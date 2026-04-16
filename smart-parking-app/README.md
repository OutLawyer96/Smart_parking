# Welcome to your Expo app 👋

This is an [Expo](https://expo.dev) project created with [`create-expo-app`](https://www.npmjs.com/package/create-expo-app).

## Get started

1. Install dependencies

   ```bash
   npm install
   ```

2. Start the app

   ```bash
   npx expo start
   ```

In the output, you'll find options to open the app in a

- [development build](https://docs.expo.dev/develop/development-builds/introduction/)
- [Android emulator](https://docs.expo.dev/workflow/android-studio-emulator/)
- [iOS simulator](https://docs.expo.dev/workflow/ios-simulator/)
- [Expo Go](https://expo.dev/go), a limited sandbox for trying out app development with Expo

You can start developing by editing the files inside the **app** directory. This project uses [file-based routing](https://docs.expo.dev/router/introduction).

## Get a fresh project

When you're ready, run:

```bash
npm run reset-project
```

This command will move the starter code to the **app-example** directory and create a blank **app** directory where you can start developing.

## Learn more

To learn more about developing your project with Expo, look at the following resources:

- [Expo documentation](https://docs.expo.dev/): Learn fundamentals, or go into advanced topics with our [guides](https://docs.expo.dev/guides).
- [Learn Expo tutorial](https://docs.expo.dev/tutorial/introduction/): Follow a step-by-step tutorial where you'll create a project that runs on Android, iOS, and the web.

## ESP32 gate control over WiFi

This app can trigger an ESP32 + servo gate when a user enters a correct PIN in the Driver flow.

### 1) Flash the ESP32 sketch

Use the Arduino sketch at `esp32/gate_servo_wifi.ino`.

- Install the Arduino library `ESP32Servo`.
- Set `WIFI_SSID` and `WIFI_PASSWORD`.
- Set your servo pin in `SERVO_PIN` (default is `18`).
- Upload to ESP32 and open Serial Monitor at `115200` baud.
- Note the printed IP, for example `10.230.113.60`.

### 2) Configure the app with ESP32 IP

Set these environment variables before starting Expo:

```bash
export EXPO_PUBLIC_ESP32_GATE_URL="http://10.230.113.60"
export EXPO_PUBLIC_ESP32_GATE_TIMEOUT_MS="7000"
export EXPO_PUBLIC_BACKEND_URL="http://192.168.1.20:8080"
export EXPO_PUBLIC_ML_MAP_URL="http://192.168.1.20:8000/api/v1/map"
npx expo start
```

`EXPO_PUBLIC_ESP32_GATE_URL` intentionally defaults to empty in code, so set it explicitly.

### 3) PIN flow behavior

- Driver enters 4-digit PIN.
- App verifies PIN with backend.
- If valid, app calls `POST /gate/open` on the ESP32.
- ESP32 opens servo (no auto-close in firmware).

### 4) Quick network checks

From your laptop (same WiFi as ESP32):

```bash
curl http://10.230.113.60/health
curl -X POST "http://10.230.113.60/gate/open"
```

If requests fail, ensure both devices are on the same LAN and no firewall blocks port `80`.

## Join the community

Join our community of developers creating universal apps.

- [Expo on GitHub](https://github.com/expo/expo): View our open source platform and contribute.
- [Discord community](https://chat.expo.dev): Chat with Expo users and ask questions.
