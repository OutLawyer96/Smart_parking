import { useParkingStore } from '../store/useParkingStore';
import type { WsEvent } from '../types/parking';

let ws: WebSocket | null = null;
let pingInterval: ReturnType<typeof setInterval> | null = null;

export function connectWebSocket(url: string, subscribeMessage?: string): void {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return;
  }
  if (ws) {
    ws.close();
    ws = null;
  }

  const socket = new WebSocket(url);
  ws = socket;

  socket.onopen = () => {
    console.log('[WS] Connected to', url);
    useParkingStore.getState().setWsConnected(true);
    useParkingStore.getState().setUpstreamConnected(true);

    if (subscribeMessage && subscribeMessage.trim()) {
      socket.send(subscribeMessage);
      pingInterval = setInterval(() => {
        if (socket.readyState === WebSocket.OPEN) {
          socket.send('ping');
        }
      }, 30_000);
    }
  };

  socket.onmessage = (event) => {
    if (event.data === 'pong') return;
    try {
      const msg = JSON.parse(event.data as string) as WsEvent;
      handleWsEvent(msg);
    } catch (err) {
      console.error('[WS] Parse error:', err);
    }
  };

  socket.onerror = () => {
    useParkingStore.getState().setWsConnected(false);
    useParkingStore.getState().setUpstreamConnected(false);
  };

  socket.onclose = (event) => {
    console.log(`[WS] Closed — code: ${event.code}`);
    useParkingStore.getState().setWsConnected(false);
    useParkingStore.getState().setUpstreamConnected(false);
    if (ws === socket) ws = null;
    if (pingInterval) {
      clearInterval(pingInterval);
      pingInterval = null;
    }
  };
}

export function disconnectWebSocket(): void {
  if (pingInterval) {
    clearInterval(pingInterval);
    pingInterval = null;
  }
  if (ws) {
    ws.close();
    ws = null;
  }
}

function handleWsEvent(msg: WsEvent): void {
  const store = useParkingStore.getState();

  switch (msg.event) {
    case 'connection_ack':
      store.setUpstreamConnected(msg.upstream_connected);
      break;

    case 'subscription_ack':
      // Initial session confirmed — slot/route from PIN verify and map from ML map endpoint are already in store
      break;

    case 'world_state':
      if (msg.cars !== undefined) {
        store.setCars(msg.cars);
        store.setCarState(msg.cars[0] ?? null);
      } else {
        store.setCarState(msg.car ?? null);
      }
      break;

    case 'map_state':
      store.setLiveMap(msg.map);
      break;

    case 'slots_state':
      store.setSlotStatuses(msg.slots);
      break;

    case 'assignments_state':
      // Assignment stream is currently informational for this UI.
      break;

    case 'parking_event':
      store.setParkingEvent(msg);
      break;

    case 'slot_update':
      store.updateSlotStatus(msg.slot_id, msg.status, msg.tracking_id);
      break;

    case 'upstream_status':
      store.setUpstreamConnected(msg.status === 'connected');
      break;

    case 'error':
      console.error('[WS]', msg.code, msg.message);
      store.setWsError(msg.message);
      break;
  }
}
