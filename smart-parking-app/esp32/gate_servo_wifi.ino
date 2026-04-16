#include <WiFi.h>
#include <WebServer.h>
#include <ESP32Servo.h>

// Replace with your WiFi credentials.
const char* WIFI_SSID = "OnePlus 10R 5G";
const char* WIFI_PASSWORD = "notfreewifi";

constexpr int SERVO_PIN = 18;
constexpr int CLOSED_ANGLE = 0;
constexpr int OPEN_ANGLE = 90;

WebServer server(80);
Servo gateServo;

int currentAngle = CLOSED_ANGLE;

void setCorsHeaders() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.sendHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  server.sendHeader("Access-Control-Allow-Headers", "Content-Type");
}

void moveGateToAngle(int angle) {
  const int clamped = constrain(angle, 0, 180);
  gateServo.write(clamped);
  currentAngle = clamped;
}

void handlePreflight() {
  setCorsHeaders();
  server.send(204);
}

void handleHealth() {
  setCorsHeaders();
  const String body = String("{") +
    "\"status\":\"ok\"," +
    "\"device\":\"esp32-gate\"," +
    "\"ip\":\"" + WiFi.localIP().toString() + "\"," +
    "\"current_angle\":" + String(currentAngle) +
    "}";
  server.send(200, "application/json", body);
}

void handleGateStatus() {
  if (server.method() != HTTP_GET && server.method() != HTTP_POST) {
    setCorsHeaders();
    server.send(405, "application/json", "{\"status\":\"error\",\"message\":\"Method not allowed\"}");
    return;
  }

  setCorsHeaders();
  const String body = String("{") +
    "\"current_angle\":" + String(currentAngle) +
    "}";
  server.send(200, "application/json", body);
}

void handleOpenGate() {
  if (server.method() != HTTP_POST) {
    setCorsHeaders();
    server.send(405, "application/json", "{\"status\":\"error\",\"message\":\"Method not allowed\"}");
    return;
  }

  if (currentAngle != CLOSED_ANGLE) {
    setCorsHeaders();
    server.send(200, "application/json", "{\"status\":\"opened\"}");
    return;
  }

  moveGateToAngle(OPEN_ANGLE);

  setCorsHeaders();
  const String body = "{\"status\":\"opened\"}";

  server.send(200, "application/json", body);
}

void handleGateClose() {
  if (server.method() != HTTP_POST) {
    setCorsHeaders();
    server.send(405, "application/json", "{\"status\":\"error\",\"message\":\"Method not allowed\"}");
    return;
  }

  if (currentAngle != OPEN_ANGLE) {
    setCorsHeaders();
    server.send(200, "application/json", "{\"status\":\"closed\"}");
    return;
  }

  moveGateToAngle(CLOSED_ANGLE);

  setCorsHeaders();
  const String body = "{\"status\":\"closed\"}";

  server.send(200, "application/json", body);
}

void setup() {
  Serial.begin(115200);

  gateServo.setPeriodHertz(50);
  gateServo.attach(SERVO_PIN, 500, 2400);
  moveGateToAngle(CLOSED_ANGLE);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print('.');
  }

  Serial.println();
  Serial.print("Connected. ESP32 IP: ");
  Serial.println(WiFi.localIP());

  server.on("/health", HTTP_GET, handleHealth);
  server.on("/health", HTTP_OPTIONS, handlePreflight);

  server.on("/gate/status", HTTP_GET, handleGateStatus);
  server.on("/gate/status", HTTP_POST, handleGateStatus);
  server.on("/gate/status", HTTP_OPTIONS, handlePreflight);

  server.on("/gate/open", HTTP_POST, handleOpenGate);
  server.on("/gate/open", HTTP_OPTIONS, handlePreflight);

  server.on("/gate/close", HTTP_POST, handleGateClose);
  server.on("/gate/close", HTTP_OPTIONS, handlePreflight);

  server.begin();
  Serial.println("HTTP server started.");
}

void loop() {
  server.handleClient();
}
