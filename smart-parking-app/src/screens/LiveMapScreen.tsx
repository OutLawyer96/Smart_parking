import React, { useEffect, useMemo, useRef } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet, Dimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import {
  Canvas, Path, Skia, Group, Circle, DashPathEffect,
} from '@shopify/react-native-skia';
import {
  useSharedValue, withSpring, withRepeat, withTiming, Easing, useDerivedValue,
} from 'react-native-reanimated';
import { router } from 'expo-router';
import { useParkingStore } from '../store/useParkingStore';
import { connectWebSocket, disconnectWebSocket } from '../services/parkingWebSocket';
import type { ParkingSlotMapData } from '../types/parking';

const { width: SW, height: SH } = Dimensions.get('window');

function buildPolygonPath(
  poly: [number, number][],
  scale: number,
  ox: number,
  oy: number,
) {
  const p = Skia.Path.Make();
  if (!poly.length) return p;
  p.moveTo(poly[0][0] * scale + ox, poly[0][1] * scale + oy);
  for (let i = 1; i < poly.length; i++) {
    p.lineTo(poly[i][0] * scale + ox, poly[i][1] * scale + oy);
  }
  p.close();
  return p;
}

const MANEUVER_LABEL: Record<string, string> = {
  straight: '↑  CONTINUE STRAIGHT',
  turn_right: '→  TURN RIGHT',
  turn_left: '←  TURN LEFT',
  park: '⊡  PARK HERE',
};

const SLOT_COLOR: Record<string, string> = {
  free: 'rgba(0,255,136,0.22)',
  occupied: 'rgba(255,68,68,0.4)',
  assigned: 'rgba(255,215,0,0.28)',
};

const LIVE_WS_URL = 'ws://127.0.0.1:8000/ws/live';

export default function LiveMapScreen() {
  const { session, wsConnected, upstreamConnected, carState, slotStatuses, parkingEvent, liveMap } =
    useParkingStore();

  const activeMap = liveMap ?? session?.map ?? null;

  // Animated values
  const carX = useSharedValue(0);
  const carY = useSharedValue(0);
  const pulseR = useSharedValue(18);
  const dashPhase = useSharedValue(0);

  // Store canvas geometry in a ref so effects can read it without stale closure
  const geoRef = useRef({ scale: 1, ox: 0, oy: 0 });

  // Canvas geometry
  const geo = useMemo(() => {
    if (!activeMap) return null;
    const { width_px, height_px } = activeMap;
    if (!width_px || !height_px) return null;
    const scale = Math.min((SW * 0.96) / width_px, (SH * 0.72) / height_px);
    const ox = (SW - width_px * scale) / 2;
    const oy = (SH - height_px * scale) / 2;
    return { scale, ox, oy };
  }, [activeMap]);

  useEffect(() => {
    if (geo) geoRef.current = geo;
  }, [geo]);

  // Connect WS on mount
  useEffect(() => {
    if (!session) { router.replace('/driver'); return; }
    connectWebSocket(LIVE_WS_URL);
    pulseR.value = withRepeat(withTiming(28, { duration: 900, easing: Easing.inOut(Easing.ease) }), -1, true);
    dashPhase.value = withRepeat(withTiming(-30, { duration: 1200, easing: Easing.linear }), -1, false);
    return () => { disconnectWebSocket(); };
  }, []);

  // Animate car
  useEffect(() => {
    if (!carState) return;
    const { scale, ox, oy } = geoRef.current;
    carX.value = withSpring(carState.cx * scale + ox, { damping: 20, stiffness: 180 });
    carY.value = withSpring(carState.cy * scale + oy, { damping: 20, stiffness: 180 });
  }, [carState?.cx, carState?.cy]);

  // Pre-build all Skia paths
  const paths = useMemo(() => {
    if (!activeMap || !geo) {
      console.log('[LiveMapScreen] Missing activeMap or geo:', { activeMap: !!activeMap, geo: !!geo });
      return null;
    }
    const { scale, ox, oy } = geo;
    const layers = activeMap.layers;
    const drivewaysLayer = Array.isArray(layers?.driveways) ? layers.driveways : [];
    const entryExitLayer = Array.isArray(layers?.entry_exit) ? layers.entry_exit : [];
    const slotsLayer = Array.isArray(layers?.parking_slots) ? layers.parking_slots : [];
    const target = session?.assigned_slot?.slot_id ?? '';
    
    console.log('[LiveMapScreen] Map layers:', {
      drivewaysCount: drivewaysLayer.length,
      entryExitCount: entryExitLayer.length,
      slotsCount: slotsLayer.length,
      target,
    });

    const driveways = drivewaysLayer.map((d) => ({
      id: d.id,
      path: buildPolygonPath(Array.isArray(d.polygon) ? d.polygon : [], scale, ox, oy),
    }));

    const entryExits = entryExitLayer.map((z) => ({
      id: z.id,
      type: z.type,
      path: buildPolygonPath(Array.isArray(z.polygon) ? z.polygon : [], scale, ox, oy),
    }));

    const slots: Array<{
      id: string;
      path: ReturnType<typeof buildPolygonPath>;
      status: ParkingSlotMapData['status'];
      isTarget: boolean;
    }> = slotsLayer.map((s) => {
      const live = slotStatuses[s.slot_id];
      return {
        id: s.slot_id,
        path: buildPolygonPath(Array.isArray(s.polygon) ? s.polygon : [], scale, ox, oy),
        status: live ? live.status : s.status,
        isTarget: s.slot_id === target,
      };
    });

    const assignedSlotPath = buildPolygonPath(
      Array.isArray(session?.assigned_slot?.polygon) ? session.assigned_slot.polygon : [],
      scale,
      ox,
      oy,
    );

    const routePath = Skia.Path.Make();
    const route = session?.route ?? [];
    if (route.length > 0) {
      const r0 = route[0];
      routePath.moveTo(r0.x * scale + ox, r0.y * scale + oy);
      for (let i = 1; i < route.length; i++) {
        const rp = route[i];
        routePath.lineTo(rp.x * scale + ox, rp.y * scale + oy);
      }
    }

    return { driveways, entryExits, slots, routePath, assignedSlotPath };
  }, [activeMap, session, slotStatuses, geo]);

  // Nearest maneuver waypoint
  const maneuver = useMemo(() => {
    if (!session) return null;
    const route = session.route ?? [];
    if (!route.length) return null;
    if (!carState) return route[0] ?? null;
    let best = route[0];
    let bestDist = Infinity;
    for (const pt of route) {
      const d = Math.hypot(pt.x - carState.cx, pt.y - carState.cy);
      if (d < bestDist) { bestDist = d; best = pt; }
    }
    return best;
  }, [session, carState]);

  const carTransform = useDerivedValue(() => [
    { translateX: carX.value },
    { translateY: carY.value },
  ]);

  const handleEnd = () => {
    disconnectWebSocket();
    useParkingStore.getState().clearSession();
    router.replace('/');
  };

  if (!session || !geo || !paths) {
    return (
      <SafeAreaView style={styles.container}>
        <Text style={styles.noSession}>No active session</Text>
        <TouchableOpacity onPress={() => router.replace('/')} style={styles.goBackBtn}>
          <Text style={styles.goBackText}>GO HOME</Text>
        </TouchableOpacity>
      </SafeAreaView>
    );
  }

  const carStatus = carState?.status ?? 'moving';

  return (
    <View style={styles.container}>
      {/* Map Canvas */}
      <Canvas style={StyleSheet.absoluteFill}>
        {/* Driveways */}
        {paths.driveways.map((d) => (
          <Path key={d.id} path={d.path} color="rgba(70,75,100,0.55)" style="fill" />
        ))}

        {/* Entry / Exit zones */}
        {paths.entryExits.map((z) => (
          <Path
            key={z.id}
            path={z.path}
            color={z.type === 'entry' ? 'rgba(0,255,136,0.2)' : 'rgba(255,100,100,0.2)'}
            style="fill"
          />
        ))}

        {/* Parking slots */}
        {paths.slots.map((s) => (
          <React.Fragment key={s.id}>
            <Path
              path={s.path}
              color={s.isTarget ? 'rgba(0,212,255,0.25)' : SLOT_COLOR[s.status] ?? SLOT_COLOR.free}
              style="fill"
            />
            <Path
              path={s.path}
              color={s.isTarget ? 'rgba(0,212,255,0.9)' : 'rgba(255,255,255,0.12)'}
              style="stroke"
              strokeWidth={s.isTarget ? 2.5 : 1}
            />
          </React.Fragment>
        ))}

        {/* Assigned slot from PIN verify payload */}
        <Path
          path={paths.assignedSlotPath}
          color="rgba(0,212,255,0.32)"
          style="fill"
        />
        <Path
          path={paths.assignedSlotPath}
          color="rgba(0,212,255,1)"
          style="stroke"
          strokeWidth={3}
        />

        {/* Navigation route */}
        <Path
          path={paths.routePath}
          style="stroke"
          strokeWidth={3.5}
          strokeCap="round"
          strokeJoin="round"
          color="#00D4FF"
        >
          <DashPathEffect intervals={[14, 9]} phase={dashPhase as unknown as number} />
        </Path>

        {/* Live car */}
        {carState && (
          <Group transform={carTransform as unknown as Parameters<typeof Group>[0]['transform']}>
            <Circle cx={0} cy={0} r={pulseR as unknown as number} color="rgba(0,212,255,0.15)" />
            <Circle cx={0} cy={0} r={9} color="#00D4FF" />
          </Group>
        )}
      </Canvas>

      {/* Header overlay */}
      <SafeAreaView pointerEvents="box-none" style={StyleSheet.absoluteFill}>
        <View style={styles.header}>
          <View style={styles.headerLeft}>
            <Text style={styles.vehicleName}>
              {session.vehicle.make} {session.vehicle.model}
            </Text>
            <Text style={styles.slotId}>Slot: {session.assigned_slot.slot_id}</Text>
          </View>
          <View style={styles.headerRight}>
            <View style={styles.statusPill}>
              <View
                style={[
                  styles.statusDot,
                  {
                    backgroundColor: wsConnected
                      ? upstreamConnected ? '#00FF88' : '#FFD700'
                      : '#FF4444',
                  },
                ]}
              />
              <Text style={styles.statusText}>
                {wsConnected ? (upstreamConnected ? 'LIVE' : 'WAIT') : 'OFFLINE'}
              </Text>
            </View>
            <TouchableOpacity onPress={handleEnd} style={styles.endBtn}>
              <Text style={styles.endBtnText}>END</Text>
            </TouchableOpacity>
          </View>
        </View>
      </SafeAreaView>

      {/* Footer — maneuver */}
      <View style={styles.footer}>
        {maneuver && (
          <Text style={styles.maneuverText}>
            {MANEUVER_LABEL[maneuver.maneuver] ?? maneuver.maneuver.toUpperCase()}
          </Text>
        )}
        {carStatus === 'parked_correct' && (
          <Text style={styles.parkedText}>✓  PARKED CORRECTLY</Text>
        )}
        {carStatus === 'parked_incorrect' && (
          <Text style={styles.warnText}>⚠  CHECK YOUR POSITION</Text>
        )}
        {parkingEvent?.type === 'parked_correct' && (
          <Text style={styles.eventText}>You have been parked in your assigned slot.</Text>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0A0A0F' },
  header: {
    margin: 16,
    marginTop: 8,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: 'rgba(10,10,20,0.85)',
    borderRadius: 18,
    padding: 14,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.08)',
  },
  headerLeft: { flex: 1 },
  headerRight: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  vehicleName: { color: '#FFFFFF', fontSize: 15, fontWeight: '700' },
  slotId: { color: '#00D4FF', fontSize: 12, fontWeight: '600', marginTop: 2 },
  statusPill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 20,
    paddingHorizontal: 10,
    paddingVertical: 5,
  },
  statusDot: { width: 7, height: 7, borderRadius: 4 },
  statusText: { color: '#FFFFFF', fontSize: 10, fontWeight: '800', letterSpacing: 1 },
  endBtn: {
    backgroundColor: 'rgba(255,68,68,0.15)',
    borderWidth: 1,
    borderColor: 'rgba(255,68,68,0.35)',
    borderRadius: 10,
    paddingHorizontal: 14,
    paddingVertical: 7,
  },
  endBtnText: { color: '#FF4444', fontSize: 11, fontWeight: '900' },
  footer: {
    position: 'absolute',
    bottom: 40,
    left: 20,
    right: 20,
    backgroundColor: 'rgba(10,10,20,0.9)',
    borderRadius: 20,
    padding: 20,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.08)',
    alignItems: 'center',
    gap: 6,
  },
  maneuverText: { color: '#00D4FF', fontSize: 17, fontWeight: '900', letterSpacing: 2 },
  parkedText: { color: '#00FF88', fontSize: 13, fontWeight: '700' },
  warnText: { color: '#FFD700', fontSize: 13, fontWeight: '700' },
  eventText: { color: 'rgba(255,255,255,0.5)', fontSize: 12 },
  noSession: { color: '#FFFFFF', fontSize: 18, textAlign: 'center', marginTop: 100 },
  goBackBtn: { alignSelf: 'center', marginTop: 20, padding: 16 },
  goBackText: { color: '#00D4FF', fontSize: 15, fontWeight: '700' },
});
