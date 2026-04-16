import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, StatusBar } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { router } from 'expo-router';

export default function HomeScreen() {
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" />
      <View style={styles.content}>
        <View style={styles.hero}>
          <Text style={styles.brand}>SMART PARKING</Text>
          <Text style={styles.tagline}>AI-POWERED PARKING SYSTEM</Text>
        </View>
        <View style={styles.cards}>
          <TouchableOpacity
            style={[styles.card, styles.cardGate]}
            onPress={() => router.push('/gate')}
            activeOpacity={0.8}
          >
            <Text style={styles.cardEmoji}>🚧</Text>
            <Text style={styles.cardTitle}>GATE ENTRY</Text>
            <Text style={styles.cardDesc}>
              Operator: register a vehicle plate and issue a parking PIN to the driver
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.card, styles.cardDriver]}
            onPress={() => router.push('/driver')}
            activeOpacity={0.8}
          >
            <Text style={styles.cardEmoji}>📍</Text>
            <Text style={styles.cardTitle}>FIND MY SPOT</Text>
            <Text style={styles.cardDesc}>
              Driver: enter your 4-digit PIN to navigate to your assigned slot
            </Text>
          </TouchableOpacity>
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0A0A0F' },
  content: { flex: 1, justifyContent: 'center', paddingHorizontal: 24 },
  hero: { alignItems: 'center', marginBottom: 48 },
  brand: { color: '#00D4FF', fontSize: 28, fontWeight: '900', letterSpacing: 6, marginBottom: 8 },
  tagline: { color: 'rgba(255,255,255,0.4)', fontSize: 11, fontWeight: '700', letterSpacing: 3 },
  cards: { gap: 16 },
  card: { borderRadius: 20, padding: 28, borderWidth: 1 },
  cardGate: { backgroundColor: 'rgba(0,212,255,0.07)', borderColor: 'rgba(0,212,255,0.25)' },
  cardDriver: { backgroundColor: 'rgba(0,255,136,0.07)', borderColor: 'rgba(0,255,136,0.25)' },
  cardEmoji: { fontSize: 36, marginBottom: 12 },
  cardTitle: { color: '#FFFFFF', fontSize: 16, fontWeight: '900', letterSpacing: 3, marginBottom: 8 },
  cardDesc: { color: 'rgba(255,255,255,0.5)', fontSize: 13, lineHeight: 20 },
});
