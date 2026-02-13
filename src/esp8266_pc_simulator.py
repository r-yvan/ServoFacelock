import time
import json
import paho.mqtt.client as mqtt

# ===================== CONFIGURATION =====================
TEAM_ID = "necromancers"  # Must match PC client team_id
MQTT_BROKER = "157.173.101.159"  # Replace with your VPS IP address
MQTT_PORT = 1883
MQTT_TOPIC = f"vision/{TEAM_ID}/movement"
MQTT_CLIENT_ID = f"pc_simulator_{TEAM_ID}"

# Mock Servo State
current_angle = 90

# ===================== MQTT CALLBACKS =====================
def on_connect(client, userdata, flags, rc):
    """MQTT connection callback"""
    if rc == 0:
        print(f"✅ Connected to MQTT broker at {MQTT_BROKER}")
        # Subscribe to movement topic
        client.subscribe(MQTT_TOPIC)
        print(f"✅ Subscribed to: {MQTT_TOPIC}")
    else:
        print(f"❌ MQTT connection failed with code: {rc}")

def on_message(client, userdata, msg):
    """Handle incoming MQTT messages"""
    global current_angle

    try:
        # Parse JSON message
        payload = msg.payload.decode('utf-8')
        data = json.loads(payload)

        status = data.get('status', 'UNKNOWN')
        confidence = data.get('confidence', 0.0)
        servo_angle = data.get('servo_angle', None)
        direction = data.get('direction', 'UNKNOWN')
        degrees_from_center = data.get('degrees_from_center', 0)

        print(f"\n📡 Received Command: {status}")
        print(f"   Confidence: {confidence:.2f} | Direction: {direction}")
        
        if servo_angle is not None:
            old_angle = current_angle
            current_angle = servo_angle
            print(f"   🔧 [SIMULATED SERVO] Moving: {old_angle:.1f}° ➔ {current_angle:.1f}°")
            print(f"   📍 Offset from center: {degrees_from_center:.1f}°")

    except Exception as e:
        print(f"❌ Error processing MQTT message: {e}")

# ===================== MAIN PROGRAM =====================
def main():
    """Main program loop"""
    print("=" * 60)
    print("      ESP8266 SERVO SIMULATOR (PC VERSION)")
    print(f"      Team ID: {TEAM_ID}")
    print("=" * 60)
    print("NOTE: This script simulates the ESP8266 on your PC.")
    print("It receives MQTT commands and logs them to this terminal.")
    print("=" * 60)

    # Initialize MQTT client
    client = mqtt.Client(MQTT_CLIENT_ID)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        print(f"Connecting to MQTT broker: {MQTT_BROKER}:{MQTT_PORT}...")
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        print(f"❌ Failed to connect to MQTT: {e}")
        return

    print("\n🚀 Simulator ready! Listening for movement commands from face_locking.py...")
    print("Press Ctrl+C to stop")

    # Start MQTT loop (blocking)
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n\nStopping simulator...")
    finally:
        client.disconnect()
        print("✅ MQTT disconnected. Program stopped.")

if __name__ == "__main__":
    main()
