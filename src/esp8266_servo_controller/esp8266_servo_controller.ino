#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <Servo.h>
#include <ArduinoJson.h>

// ===================== CONFIGURATION =====================

const char* TEAM_ID = "necromancers";
const char* MQTT_BROKER = "157.173.101.159";  // Your VPS MQTT broker
const int MQTT_PORT = 1883;
const char* MQTT_TOPIC = "vision/necromancers/movement";
const char* MQTT_CLIENT_ID = "esp8266_necromancers";

// WiFi configuration
const char* WIFI_SSID = "RCA";
const char* WIFI_PASSWORD = "@RcaNyabihu2023";

// Servo configuration
const int SERVO_PIN = D4;  // GPIO2 on NodeMCU
Servo myservo;
int current_angle = 90;
int target_angle = 90;

// ===================== GLOBAL VARIABLES =====================
WiFiClient espClient;
PubSubClient client(espClient);

// ===================== MQTT SETUP =====================
void setup_mqtt() {
  client.setServer(MQTT_BROKER, MQTT_PORT);
  client.setCallback(callback);
}

// ===================== WIFI FUNCTIONS =====================
bool connect_wifi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(WIFI_SSID);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int timeout = 20;
  while (WiFi.status() != WL_CONNECTED && timeout > 0) {
    delay(1000);
    Serial.print(".");
    timeout--;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✓ WiFi connected!");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
    return true;
  } else {
    Serial.println("\n✗ WiFi connection failed");
    return false;
  }
}

// ===================== MQTT FUNCTIONS =====================
bool connect_mqtt() {
  if (!client.connected()) {
    Serial.print("Connecting to MQTT broker: ");
    Serial.print(MQTT_BROKER);
    Serial.print(":");
    Serial.println(MQTT_PORT);

    if (client.connect(MQTT_CLIENT_ID)) {
      Serial.println("✓ Connected to MQTT broker");
      client.subscribe(MQTT_TOPIC);
      Serial.print("✓ Subscribed to: ");
      Serial.println(MQTT_TOPIC);
      return true;
    } else {
      Serial.print("✗ MQTT connection failed, rc=");
      Serial.println(client.state());
      return false;
    }
  }
  return true;
}

void callback(char* topic, byte* payload, unsigned int length) {
  // Parse JSON message
  String message = "";
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }

  Serial.print("📡 Received message: ");
  Serial.println(message);

  // Parse JSON
  DynamicJsonDocument doc(1024);
  DeserializationError error = deserializeJson(doc, message);

  if (error) {
    Serial.print("✗ JSON parsing failed: ");
    Serial.println(error.c_str());
    return;
  }

  // Extract data
  String status = doc["status"] | "UNKNOWN";
  float confidence = doc["confidence"] | 0.0;
  int servo_angle = doc["servo_angle"] | 90;
  String direction = doc["direction"] | "UNKNOWN";
  float degrees_from_center = doc["degrees_from_center"] | 0.0;

  Serial.print("📡 Status: ");
  Serial.print(status);
  Serial.print(" | Confidence: ");
  Serial.print(confidence);
  Serial.print(" | Direction: ");
  Serial.println(direction);

  // Move servo based on command
  if (servo_angle >= 0 && servo_angle <= 180) {
    move_servo(servo_angle);
    Serial.print("📍 Face position: ");
    Serial.print(degrees_from_center);
    Serial.println("° from center");
  }
}

// ===================== SERVO FUNCTIONS =====================
void move_servo(int angle) {
  // Clamp angle to valid range
  if (angle < 0) angle = 0;
  if (angle > 180) angle = 180;

  target_angle = angle;

  // Smooth movement (gradual transition)
  int steps = 10;
  int step_delay = 20;  // 20ms between steps

  for (int i = 0; i <= steps; i++) {
    int intermediate_angle = current_angle + (target_angle - current_angle) * i / steps;
    myservo.write(intermediate_angle);
    delay(step_delay);
  }

  current_angle = target_angle;
  Serial.print("🔧 Servo moved to ");
  Serial.print(angle);
  Serial.println("°");
}

// ===================== SETUP =====================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("==================================================");
  Serial.println("ESP8266 Servo Controller - Arduino Version");
  Serial.print("Team ID: ");
  Serial.println(TEAM_ID);
  Serial.println("==================================================");

  // Initialize servo
  myservo.attach(SERVO_PIN);
  myservo.write(90);  // Start at center position
  Serial.println("✓ Servo initialized");

  // Connect to WiFi
  if (!connect_wifi()) {
    Serial.println("Failed to connect to WiFi. Restarting...");
    ESP.restart();
  }

  // Setup MQTT
  setup_mqtt();

  // Connect to MQTT
  if (!connect_mqtt()) {
    Serial.println("Failed to connect to MQTT. Restarting...");
    ESP.restart();
  }

  Serial.println("\n✓ System ready! Waiting for movement commands...");
}

// ===================== MAIN LOOP =====================
void loop() {
  if (!client.connected()) {
    connect_mqtt();
  }
  client.loop();
  delay(100);  // Small delay to prevent overwhelming
}
