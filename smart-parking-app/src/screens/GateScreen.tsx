import React, { useState } from 'react';
import {
  View, Text, TextInput, TouchableOpacity, StyleSheet,
  StatusBar, ActivityIndicator, ScrollView, Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { router } from 'expo-router';
import { registerVehicle, sendCodeToLcd } from '../services/parkingApi';
import type { RegisterSuccessResponse, RegisterErrorResponse } from '../types/parking';

type Result = { ok: true; data: RegisterSuccessResponse } | { ok: false; data: RegisterErrorResponse };

export default function GateScreen() {
  const [plate, setPlate] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Result | null>(null);

  const handleRegister = async () => {
    const trimmed = plate.trim().toUpperCase();
    if (!trimmed) { Alert.alert('Enter a plate number'); return; }
    setLoading(true);
    setResult(null);
    try {
      const res = await registerVehicle(trimmed);
      if (res.status === 'assigned') {
        try {
          await sendCodeToLcd(res.pin);
        } catch (lcdError) {
          Alert.alert('PIN generated, LCD not updated', String(lcdError));
        }
        setResult({ ok: true, data: res as RegisterSuccessResponse });
      } else {
        setResult({ ok: false, data: res as RegisterErrorResponse });
      }
    } catch (err) {
      Alert.alert('Network error', String(err));
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => { setPlate(''); setResult(null); };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" />
      <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()} style={styles.backBtn}>
            <Text style={styles.backBtnText}>← BACK</Text>
          </TouchableOpacity>
          <Text style={styles.title}>GATE ENTRY</Text>
          <Text style={styles.subtitle}>OPERATOR PANEL</Text>
        </View>

        {/* Plate input */}
        <View style={styles.inputSection}>
          <Text style={styles.label}>VEHICLE PLATE NUMBER</Text>
          <TextInput
            style={styles.input}
            placeholder="e.g. DL8CAF5031"
            placeholderTextColor="rgba(255,255,255,0.25)"
            value={plate}
            onChangeText={(t) => setPlate(t.toUpperCase())}
            autoCapitalize="characters"
            editable={!loading}
          />
          <TouchableOpacity
            style={[styles.registerBtn, loading && styles.registerBtnDisabled]}
            onPress={handleRegister}
            disabled={loading}
            activeOpacity={0.8}
          >
            {loading ? (
              <ActivityIndicator color="#0A0A0F" />
            ) : (
              <Text style={styles.registerBtnText}>REGISTER VEHICLE</Text>
            )}
          </TouchableOpacity>
        </View>

        {/* Result */}
        {/* {result && (
          result.ok ? (
            <SuccessCard data={result.data} onReset={handleReset} />
          ) : (
            <ErrorCard data={result.data} onReset={handleReset} />
          )
        )} */}
      </ScrollView>
    </SafeAreaView>
  );
}

function SuccessCard({ data, onReset }: { data: RegisterSuccessResponse; onReset: () => void }) {
  return (
    <View style={styles.resultCard}>
      <Text style={styles.resultLabel}>VEHICLE REGISTERED</Text>

      {/* Vehicle info */}
      <View style={styles.infoRow}>
        <Text style={styles.infoKey}>VEHICLE</Text>
        <Text style={styles.infoVal}>{data.vehicle.make} {data.vehicle.model}</Text>
      </View>
      <View style={styles.infoRow}>
        <Text style={styles.infoKey}>PLATE</Text>
        <Text style={styles.infoVal}>{data.plate_number}</Text>
      </View>
      <View style={styles.infoRow}>
        <Text style={styles.infoKey}>SLOT</Text>
        <Text style={[styles.infoVal, { color: '#00D4FF' }]}>{data.assigned_slot.slot_id}</Text>
      </View>
      <View style={styles.infoRow}>
        <Text style={styles.infoKey}>ZONE</Text>
        <Text style={styles.infoVal}>{data.assigned_slot.zone}</Text>
      </View>

      {/* PIN display */}
      <View style={styles.pinContainer}>
        <Text style={styles.pinLabel}>DRIVER PIN</Text>
        <View style={styles.pinDigits}>
          {data.pin.split('').map((d, i) => (
            <View key={i} style={styles.pinDigitBox}>
              <Text style={styles.pinDigit}>{d}</Text>
            </View>
          ))}
        </View>
        <Text style={styles.pinExpiry}>Expires in {data.pin_expires_in_minutes} minutes</Text>
      </View>

      <TouchableOpacity style={styles.resetBtn} onPress={onReset} activeOpacity={0.8}>
        <Text style={styles.resetBtnText}>REGISTER ANOTHER VEHICLE</Text>
      </TouchableOpacity>
    </View>
  );
}

function ErrorCard({ data, onReset }: { data: RegisterErrorResponse; onReset: () => void }) {
  const messages: Record<string, string> = {
    vehicle_not_found: 'This plate is not in the database. Please add the vehicle first.',
    no_slots_available: 'The parking lot is currently full. No slots available.',
    ai_not_ready: 'The AI system is still warming up. Please try again in a moment.',
    error: 'An unexpected error occurred. Please try again.',
  };

  return (
    <View style={[styles.resultCard, styles.resultCardError]}>
      <Text style={[styles.resultLabel, { color: '#FF4444' }]}>REGISTRATION FAILED</Text>
      <Text style={styles.errorMessage}>{messages[data.status] ?? data.message}</Text>
      <View style={styles.infoRow}>
        <Text style={styles.infoKey}>PLATE</Text>
        <Text style={styles.infoVal}>{data.plate_number}</Text>
      </View>
      <TouchableOpacity style={styles.resetBtn} onPress={onReset} activeOpacity={0.8}>
        <Text style={styles.resetBtnText}>TRY AGAIN</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0A0A0F' },
  scroll: { padding: 24, paddingBottom: 60 },
  header: { marginBottom: 36 },
  backBtn: { marginBottom: 20 },
  backBtnText: { color: '#00D4FF', fontSize: 13, fontWeight: '700', letterSpacing: 2 },
  title: { color: '#FFFFFF', fontSize: 26, fontWeight: '900', letterSpacing: 4 },
  subtitle: { color: 'rgba(255,255,255,0.35)', fontSize: 11, fontWeight: '700', letterSpacing: 3, marginTop: 4 },
  inputSection: { marginBottom: 28 },
  label: { color: 'rgba(255,255,255,0.5)', fontSize: 11, fontWeight: '700', letterSpacing: 2, marginBottom: 10 },
  input: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.12)',
    borderRadius: 14,
    color: '#FFFFFF',
    fontSize: 20,
    fontWeight: '700',
    letterSpacing: 3,
    padding: 18,
    marginBottom: 14,
  },
  registerBtn: {
    backgroundColor: '#00D4FF',
    borderRadius: 14,
    paddingVertical: 16,
    alignItems: 'center',
  },
  registerBtnDisabled: { opacity: 0.5 },
  registerBtnText: { color: '#0A0A0F', fontSize: 15, fontWeight: '900', letterSpacing: 2 },
  resultCard: {
    backgroundColor: 'rgba(0,255,136,0.06)',
    borderWidth: 1,
    borderColor: 'rgba(0,255,136,0.2)',
    borderRadius: 20,
    padding: 24,
  },
  resultCardError: {
    backgroundColor: 'rgba(255,68,68,0.06)',
    borderColor: 'rgba(255,68,68,0.2)',
  },
  resultLabel: { color: '#00FF88', fontSize: 11, fontWeight: '900', letterSpacing: 3, marginBottom: 20 },
  infoRow: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 10 },
  infoKey: { color: 'rgba(255,255,255,0.4)', fontSize: 11, fontWeight: '700', letterSpacing: 1 },
  infoVal: { color: '#FFFFFF', fontSize: 13, fontWeight: '600' },
  pinContainer: { marginTop: 28, marginBottom: 8, alignItems: 'center' },
  pinLabel: { color: 'rgba(255,255,255,0.4)', fontSize: 11, fontWeight: '700', letterSpacing: 3, marginBottom: 16 },
  pinDigits: { flexDirection: 'row', gap: 12, marginBottom: 12 },
  pinDigitBox: {
    width: 64,
    height: 80,
    borderRadius: 14,
    backgroundColor: 'rgba(0,212,255,0.12)',
    borderWidth: 2,
    borderColor: 'rgba(0,212,255,0.4)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  pinDigit: { color: '#00D4FF', fontSize: 36, fontWeight: '900' },
  pinExpiry: { color: 'rgba(255,255,255,0.35)', fontSize: 11, marginTop: 4 },
  resetBtn: {
    marginTop: 20,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.15)',
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: 'center',
  },
  resetBtnText: { color: 'rgba(255,255,255,0.6)', fontSize: 12, fontWeight: '700', letterSpacing: 2 },
  errorMessage: { color: 'rgba(255,255,255,0.7)', fontSize: 14, lineHeight: 22, marginBottom: 16 },
});
