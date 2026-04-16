import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';

export default function RootLayout() {
  return (
    <>
      <Stack
        screenOptions={{
          headerShown: false,
          contentStyle: { backgroundColor: '#0A0A0F' },
          animation: 'slide_from_right',
        }}
      >
        <Stack.Screen name="index" />
        <Stack.Screen name="gate" />
        <Stack.Screen name="driver" />
        <Stack.Screen name="live" options={{ animation: 'fade' }} />
      </Stack>
      <StatusBar style="light" />
    </>
  );
}
