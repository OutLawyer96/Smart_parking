import type { RegisterResponse, PinVerifyResponse, ParkingMap } from '../types/parking';

const BASE_URL = process.env.EXPO_PUBLIC_BACKEND_URL ?? 'http://localhost:8080';
const ML_MAP_URL = process.env.EXPO_PUBLIC_ML_MAP_URL ?? 'http://127.0.0.1:8000/api/v1/map';
const ESP32_GATE_URL = process.env.EXPO_PUBLIC_ESP32_GATE_URL ?? 'http://10.54.215.60';
const ESP32_GATE_REQUEST_TIMEOUT_MS = Number(process.env.EXPO_PUBLIC_ESP32_GATE_TIMEOUT_MS ?? '3000');

export type GateStatusResponse = {
  current_angle: number;
};

export type GateOpenResponse = {
  status: 'opened';
};

export type GateCloseResponse = {
  status: 'closed';
};

export type LcdCodeResponse = {
  status: 'code_displayed';
};

export async function registerVehicle(plateNumber: string): Promise<RegisterResponse> {
  const response = await fetch(`${BASE_URL}/api/parking/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ plate_number: plateNumber }),
  });
  return response.json() as Promise<RegisterResponse>;
}

export async function verifyPin(pin: string): Promise<PinVerifyResponse> {
  const response = await fetch(`${BASE_URL}/api/parking/pin/verify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pin }),
  });
  return response.json() as Promise<PinVerifyResponse>;
}

export async function getGateStatus(): Promise<GateStatusResponse> {
  if (!ESP32_GATE_URL.trim()) {
    throw new Error('Set EXPO_PUBLIC_ESP32_GATE_URL before checking gate status.');
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), ESP32_GATE_REQUEST_TIMEOUT_MS);

  try {
    const endpoint = `${ESP32_GATE_URL}/gate/status`;
    const response = await fetch(endpoint, {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(`ESP32 status request failed (${response.status}).`);
    }

    const body = (await response.json()) as Partial<GateStatusResponse>;
    if (typeof body.current_angle !== 'number') {
      throw new Error('ESP32 status returned an invalid response.');
    }

    return { current_angle: body.current_angle };
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error('ESP32 status request timed out. Check WiFi and IP address.');
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

export async function triggerGateOpen(): Promise<GateOpenResponse> {
  if (!ESP32_GATE_URL.trim()) {
    throw new Error('Set EXPO_PUBLIC_ESP32_GATE_URL before triggering gate.');
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), ESP32_GATE_REQUEST_TIMEOUT_MS);

  try {
    const endpoint = `${ESP32_GATE_URL}/gate/open`;
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { Accept: 'application/json' },
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(`ESP32 gate request failed (${response.status}).`);
    }

    const body = (await response.json()) as Partial<GateOpenResponse> & { message?: string };
    if (body.status !== 'opened') {
      throw new Error(body.message ?? 'ESP32 gate returned an invalid response.');
    }

    return { status: 'opened' };
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error('ESP32 gate request timed out. Check WiFi and IP address.');
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

export async function triggerGateClose(): Promise<GateCloseResponse> {
  if (!ESP32_GATE_URL.trim()) {
    throw new Error('Set EXPO_PUBLIC_ESP32_GATE_URL before closing gate.');
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), ESP32_GATE_REQUEST_TIMEOUT_MS);

  try {
    const endpoint = `${ESP32_GATE_URL}/gate/close`;
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { Accept: 'application/json' },
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(`ESP32 gate close request failed (${response.status}).`);
    }

    const body = (await response.json()) as Partial<GateCloseResponse> & { message?: string };
    if (body.status !== 'closed') {
      throw new Error(body.message ?? 'ESP32 gate close returned an invalid response.');
    }

    return { status: 'closed' };
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error('ESP32 gate close request timed out. Check WiFi and IP address.');
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

export async function sendCodeToLcd(code: string): Promise<LcdCodeResponse> {
  if (!ESP32_GATE_URL.trim()) {
    throw new Error('Set EXPO_PUBLIC_ESP32_GATE_URL before sending code to LCD.');
  }

  if (!/^\d{4}$/.test(code)) {
    throw new Error('LCD code must be exactly 4 digits.');
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), ESP32_GATE_REQUEST_TIMEOUT_MS);

  try {
    const endpoint = `${ESP32_GATE_URL}/lcd/code`;
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: `code=${encodeURIComponent(code)}`,
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(`ESP32 LCD request failed (${response.status}).`);
    }

    const body = (await response.json()) as Partial<LcdCodeResponse> & { error?: string; message?: string };
    if (body.status !== 'code_displayed') {
      throw new Error(body.error ?? body.message ?? 'ESP32 LCD returned an invalid response.');
    }

    return { status: 'code_displayed' };
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error('ESP32 LCD request timed out. Check WiFi and IP address.');
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

const wait = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

export async function runGateOpenCloseCycle(): Promise<void> {
  const openBefore = await getGateStatus();
  if (openBefore.current_angle !== 0) {
    throw new Error('Gate is not closed (angle is not 0).');
  }

  await triggerGateOpen();
  await wait(3000);

  const openAfter = await getGateStatus();
  if (openAfter.current_angle !== 90) {
    throw new Error('Gate is not open (angle is not 90) before close attempt.');
  }

  await triggerGateClose();
}

type MapApiResponse = {
  map?: unknown;
};

function normalizeParkingMap(raw: unknown): ParkingMap | null {
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
}

export async function fetchParkingMapFromMl(): Promise<ParkingMap> {
  console.log('[parkingApi] Fetching map from:', ML_MAP_URL);
  
  const response = await fetch(ML_MAP_URL, {
    method: 'GET',
    headers: { Accept: 'application/json' },
  });

  if (!response.ok) {
    throw new Error(`Map API request failed (${response.status}).`);
  }

  const body = (await response.json()) as MapApiResponse;
  console.log('[parkingApi] Raw map response:', body);
  
  const normalizedMap = normalizeParkingMap(body.map);
  console.log('[parkingApi] Normalized map:', normalizedMap);
  console.log('[parkingApi] Parking slots count:', normalizedMap?.layers.parking_slots.length ?? 0);

  if (!normalizedMap) {
    throw new Error('Map API returned an invalid payload.');
  }

  return normalizedMap;
}