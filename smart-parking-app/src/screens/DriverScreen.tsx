import React, { useState } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet, StatusBar, ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { router } from 'expo-router';
import { fetchParkingMapFromMl, runGateOpenCloseCycle, verifyPin } from '../services/parkingApi';
import { useParkingStore } from '../store/useParkingStore';
import type { ParkingSession, ParkingMap } from '../types/parking';

const KEYS = ['1','2','3','4','5','6','7','8','9','','0','⌫'];

export default function DriverScreen() {
  const [pin, setPin] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const setSession = useParkingStore((s) => s.setSession);

  const handleKey = (key: string) => {
    if (key === '⌫') {
      setPin((p) => p.slice(0, -1));
      setError(null);
    } else if (pin.length < 4) {
      const next = pin + key;
      setPin(next);
      setError(null);
      if (next.length === 4) handleVerify(next);
    }
  };

  const handleVerify = async (code: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await verifyPin(code);
      if ('error' in res) {
        setError(res.message);
        setPin('');
      } else {
        // Helper function to normalize map from PIN response
        const normalizeMapFromPin = (raw: unknown): ParkingMap | null => {
          if (!raw || typeof raw !== 'object') return null;
          const candidate = raw as Partial<ParkingMap> & {
            layers?: Partial<ParkingMap['layers']>;
          };

          if (typeof candidate.width_px !== 'number' || typeof candidate.height_px !== 'number') {
            return null;
          }

          const layers: Partial<ParkingMap['layers']> = candidate.layers ?? {};

          return {
            width_px: candidate.width_px,
            height_px: candidate.height_px,
            scale_m_per_px: typeof candidate.scale_m_per_px === 'number' ? candidate.scale_m_per_px : 0,
            layers: {
              parking_slots: Array.isArray(layers.parking_slots) ? layers.parking_slots : [],
              restricted_zones: Array.isArray(layers.restricted_zones) ? layers.restricted_zones : [],
              entry_exit: Array.isArray(layers.entry_exit) ? layers.entry_exit : [],
              driveways: Array.isArray(layers.driveways) ? layers.driveways : [],
            },
          };
        };

        // Try to use map from PIN response first
        let map = normalizeMapFromPin(res.map);
        
        // If PIN response didn't have a usable map, try to fetch from ML endpoint
        if (!map || map.layers.parking_slots.length === 0) {
          console.log('[DriverScreen] No map from PIN response, fetching from ML endpoint...');
          try {
            map = await fetchParkingMapFromMl();
          } catch (mlErr) {
            console.warn('[DriverScreen] Failed to fetch from ML endpoint, using PIN map:', mlErr);
            // If ML fails but PIN response had a map, use that anyway
            map = normalizeMapFromPin(res.map) ?? {
              width_px: 440,
              height_px: 499,
              scale_m_per_px: 0.001128,
              layers: {
                parking_slots: [],
                restricted_zones: [],
                entry_exit: [],
                driveways: [],
              },
            };
          }
        }

        // Also run gate cycle in parallel
        try {
          await runGateOpenCloseCycle();
        } catch (gateErr) {
          console.warn('[DriverScreen] Gate cycle failed:', gateErr);
          // Don't fail if gate cycle has issues
        }

        if (!map) {
          throw new Error('Failed to load parking map.');
        }

        const session: ParkingSession = { ...res, map };
        setSession(session);
        router.replace('/live');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Network error. Please check your connection.');
      setPin('');
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" />
      <View style={styles.content}>
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()} style={styles.backBtn}>
            <Text style={styles.backBtnText}>← BACK</Text>
          </TouchableOpacity>
          <Text style={styles.title}>ENTER YOUR PIN</Text>
          <Text style={styles.subtitle}>4-DIGIT CODE FROM GATE DISPLAY</Text>
        </View>

        {/* PIN dots */}
        <View style={styles.dotsRow}>
          {[0,1,2,3].map((i) => (
            <View
              key={i}
              style={[
                styles.dot,
                i < pin.length && styles.dotFilled,
                error ? styles.dotError : null,
              ]}
            />
          ))}
        </View>

        {/* Error */}
        {error && <Text style={styles.errorText}>{error}</Text>}

        {/* Loading */}
        {loading && <ActivityIndicator color="#00D4FF" style={{ marginBottom: 20 }} />}

        {/* Numpad */}
        <View style={styles.numpad}>
          {KEYS.map((key, idx) => (
            key === '' ? (
              <View key={idx} style={styles.keyEmpty} />
            ) : (
              <TouchableOpacity
                key={idx}
                style={[styles.key, key === '⌫' && styles.keyDelete]}
                onPress={() => !loading && handleKey(key)}
                activeOpacity={0.7}
              >
                <Text style={[styles.keyText, key === '⌫' && styles.keyDeleteText]}>{key}</Text>
              </TouchableOpacity>
            )
          ))}
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0A0A0F' },
  content: { flex: 1, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 32 },
  header: { alignItems: 'center', marginBottom: 40, width: '100%' },
  backBtn: { alignSelf: 'flex-start', marginBottom: 24 },
  backBtnText: { color: '#00D4FF', fontSize: 13, fontWeight: '700', letterSpacing: 2 },
  title: { color: '#FFFFFF', fontSize: 22, fontWeight: '900', letterSpacing: 4, marginBottom: 6 },
  subtitle: { color: 'rgba(255,255,255,0.35)', fontSize: 10, fontWeight: '700', letterSpacing: 2 },
  dotsRow: { flexDirection: 'row', gap: 18, marginBottom: 16 },
  dot: {
    width: 20, height: 20, borderRadius: 10,
    borderWidth: 2, borderColor: 'rgba(255,255,255,0.2)',
    backgroundColor: 'transparent',
  },
  dotFilled: { backgroundColor: '#00D4FF', borderColor: '#00D4FF' },
  dotError: { borderColor: '#FF4444', backgroundColor: 'rgba(255,68,68,0.3)' },
  errorText: { color: '#FF4444', fontSize: 13, marginBottom: 16, textAlign: 'center' },
  numpad: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    width: 280,
    gap: 12,
    marginTop: 24,
  },
  key: {
    width: 80, height: 80, borderRadius: 40,
    backgroundColor: 'rgba(255,255,255,0.06)',
    borderWidth: 1, borderColor: 'rgba(255,255,255,0.1)',
    justifyContent: 'center', alignItems: 'center',
  },
  keyDelete: {
    backgroundColor: 'rgba(255,68,68,0.08)',
    borderColor: 'rgba(255,68,68,0.2)',
  },
  keyEmpty: { width: 80, height: 80 },
  keyText: { color: '#FFFFFF', fontSize: 24, fontWeight: '600' },
  keyDeleteText: { color: '#FF4444', fontSize: 22 },
});
