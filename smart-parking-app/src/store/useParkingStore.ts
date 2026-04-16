import { create } from 'zustand';
import type { ParkingSession, CarState, ParkingEventMsg, SlotStatus } from '../types/parking';
import type { ParkingMap, SlotsStateItem } from '../types/parking';

type ParkingStoreState = {
  session: ParkingSession | null;
  wsConnected: boolean;
  upstreamConnected: boolean;
  carState: CarState | null;
  cars: CarState[];
  liveMap: ParkingMap | null;
  slotStatuses: Record<string, SlotStatus>;
  parkingEvent: ParkingEventMsg | null;
  wsError: string | null;

  setSession: (session: ParkingSession) => void;
  clearSession: () => void;
  setWsConnected: (connected: boolean) => void;
  setUpstreamConnected: (connected: boolean) => void;
  setCarState: (car: CarState | null) => void;
  setCars: (cars: CarState[]) => void;
  setLiveMap: (map: ParkingMap | null) => void;
  setSlotStatuses: (slots: SlotsStateItem[]) => void;
  updateSlotStatus: (slotId: string, status: SlotStatus['status'], trackingId: number | null) => void;
  setParkingEvent: (event: ParkingEventMsg) => void;
  setWsError: (error: string | null) => void;
};

export const useParkingStore = create<ParkingStoreState>((set) => ({
  session: null,
  wsConnected: false,
  upstreamConnected: false,
  carState: null,
  cars: [],
  liveMap: null,
  slotStatuses: {},
  parkingEvent: null,
  wsError: null,

  setSession: (session) => set({ session }),
  clearSession: () =>
    set({
      session: null,
      wsConnected: false,
      upstreamConnected: false,
      carState: null,
      cars: [],
      liveMap: null,
      slotStatuses: {},
      parkingEvent: null,
      wsError: null,
    }),
  setWsConnected: (connected) => set({ wsConnected: connected }),
  setUpstreamConnected: (connected) => set({ upstreamConnected: connected }),
  setCarState: (car) => set({ carState: car }),
  setCars: (cars) => set({ cars }),
  setLiveMap: (liveMap) => set({ liveMap }),
  setSlotStatuses: (slots) =>
    set(() => {
      const next: Record<string, SlotStatus> = {};
      for (const slot of slots) {
        next[slot.slot_id] = { status: slot.status, tracking_id: slot.tracking_id };
      }
      return { slotStatuses: next };
    }),
  updateSlotStatus: (slotId, status, tracking_id) =>
    set((state) => ({
      slotStatuses: { ...state.slotStatuses, [slotId]: { status, tracking_id } },
    })),
  setParkingEvent: (event) => set({ parkingEvent: event }),
  setWsError: (error) => set({ wsError: error }),
}));
