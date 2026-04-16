export interface Vehicle {
  plate_number: string;
  make: string;
  model: string;
  length_m: number;
  width_m: number;
}

export interface AssignedSlot {
  slot_id: string;
  cx: number;
  cy: number;
  polygon: [number, number][];
  zone: string;
  distance_from_exit_m: number;
}

export type Maneuver = 'straight' | 'turn_right' | 'turn_left' | 'park';

export interface RoutePoint {
  x: number;
  y: number;
  maneuver: Maneuver;
}

export interface ParkingSlotMapData {
  slot_id: string;
  cx: number;
  cy: number;
  polygon: [number, number][];
  status: 'free' | 'occupied' | 'assigned';
}

export interface EntryExitZone {
  id: string;
  type: string;
  polygon: [number, number][];
}

export interface Driveway {
  id: string;
  polygon: [number, number][];
}

export interface MapLayers {
  parking_slots: ParkingSlotMapData[];
  restricted_zones: unknown[];
  entry_exit: EntryExitZone[];
  driveways: Driveway[];
}

export interface ParkingMap {
  width_px: number;
  height_px: number;
  scale_m_per_px: number;
  layers: MapLayers;
}

// ── Register ────────────────────────────────────────────────────────────────

export interface RegisterSuccessResponse {
  plate_number: string;
  tracking_id: number;
  status: 'assigned';
  pin: string;
  pin_expires_in_minutes: number;
  vehicle: Vehicle;
  assigned_slot: AssignedSlot;
  route: RoutePoint[];
}

export interface RegisterErrorResponse {
  plate_number: string;
  tracking_id?: number;
  status: 'vehicle_not_found' | 'no_slots_available' | 'ai_not_ready' | 'error';
  message: string;
  vehicle?: Vehicle;
}

export type RegisterResponse = RegisterSuccessResponse | RegisterErrorResponse;

// ── PIN verify ──────────────────────────────────────────────────────────────

export interface PinVerifySuccessResponse {
  pin: string;
  tracking_id: number;
  plate_number: string;
  expires_at: string;
  vehicle: Vehicle;
  assigned_slot: AssignedSlot;
  route: RoutePoint[];
  // Backend may return a non-renderable placeholder map object; frontend fetches map separately.
  map: unknown;
  websocket: { url: string; subscribe_message: string };
}

export interface PinVerifyErrorResponse {
  error: 'MISSING_PIN' | 'INVALID_PIN';
  message: string;
}

export type PinVerifyResponse = PinVerifySuccessResponse | PinVerifyErrorResponse;

export type ParkingSession = Omit<PinVerifySuccessResponse, 'map'> & {
  map: ParkingMap;
};

// ── Live state ──────────────────────────────────────────────────────────────

export interface CarState {
  tracking_id: number;
  cx: number;
  cy: number;
  polygon: [number, number][];
  heading_deg: number;
  status: 'moving' | 'parked_correct' | 'parked_incorrect';
  assigned_slot: string | null;
  parked_duration_seconds: number | null;
}

export interface ParkingEventMsg {
  event: 'parking_event';
  tracking_id: number;
  type: 'parked_correct' | 'parked_incorrect' | 'unparked';
  slot_id: string;
  timestamp: string;
}

export interface SlotStatus {
  status: 'free' | 'occupied' | 'assigned';
  tracking_id: number | null;
}

export interface SlotsStateItem {
  slot_id: string;
  status: SlotStatus['status'];
  tracking_id: number | null;
}

// ── WebSocket events ─────────────────────────────────────────────────────────

export type WsEvent =
  | { event: 'connection_ack'; status: string; upstream_connected: boolean; message: string }
  | {
      event: 'subscription_ack';
      tracking_id: number;
      plate_number: string;
      vehicle: Vehicle;
      assigned_slot: AssignedSlot;
      route: RoutePoint[];
      map: ParkingMap;
      message: string;
    }
  | {
      event: 'world_state';
      timestamp: string;
      layout_version?: number;
      tracking_id?: number;
      car?: CarState | null;
      cars?: CarState[];
    }
  | { event: 'map_state'; layout_version: number; map: ParkingMap }
  | { event: 'slots_state'; layout_version: number; slots: SlotsStateItem[] }
  | { event: 'assignments_state'; layout_version: number; assignments: unknown[] }
  | ParkingEventMsg
  | { event: 'slot_update'; slot_id: string; status: 'free' | 'occupied' | 'assigned'; tracking_id: number | null }
  | { event: 'upstream_status'; status: 'connected' | 'disconnected' }
  | { event: 'error'; code: string; message: string };
