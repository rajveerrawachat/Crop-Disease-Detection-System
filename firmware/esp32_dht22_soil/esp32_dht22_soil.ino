/*
 * ============================================================================
 * ESP32 Crop Environmental Telemetry Firmware
 *
 * HARDWARE CONFIGURATION & WIRING:
 * 1. DHT22 Digital Temperature & Humidity Sensor:
 *    - VCC  -> ESP32 3.3V (Do NOT connect 5V to GPIO!)
 *    - GND  -> ESP32 GND
 *    - DATA -> ESP32 GPIO 4 (Add 10k pull-up resistor to 3.3V if bare sensor)
 *
 * 2. HW-080 / Capacitive Soil Moisture Sensor:
 *    - VCC  -> ESP32 3.3V (CRITICAL: ESP32 ADC inputs are 3.3V MAX. Never use 5V!)
 *    - GND  -> ESP32 GND
 *    - AO   -> ESP32 GPIO 34 (ADC1 Channel 6, input-only pin)
 *
 * SERIAL PROTOCOL:
 *    - Baud Rate: 115200
 *    - Format: One newline-terminated JSON object per reading interval
 *    - Normal Payload: {"temperature":25.4,"humidity":60.1,"soil_moisture":45.2}
 *    - Error Payload:  {"error":"DHT22_read_failed","soil_moisture":45.2}
 *
 * REQUIRED ARDUINO LIBRARIES:
 *    - "DHT sensor library" by Adafruit (Install via Arduino Library Manager)
 *    - "Adafruit Unified Sensor" by Adafruit
 * ============================================================================
 */

#include <Arduino.h>
#include "DHT.h"

// ----------------------------------------------------------------------------
// PIN ASSIGNMENTS & SENSOR TYPE
// ----------------------------------------------------------------------------
const int DHT_PIN = 4;        // GPIO 4 connected to DHT22 DATA
#define DHT_TYPE DHT22        // Sensor type DHT22 (AM2302)

const int SOIL_PIN = 34;      // GPIO 34 connected to Soil Moisture Analog Out (AO)

// ----------------------------------------------------------------------------
// SOIL MOISTURE CALIBRATION CONSTANTS
// ----------------------------------------------------------------------------
// ESP32 ADC is 12-bit (raw integer values 0 to 4095).
// Most soil sensors output a higher voltage in dry conditions and lower in wet:
//   - Dry Air / Dry Soil Reference (0% Moisture): ~3200 - 3800
//   - Saturated Soil / Water Reference (100% Moisture): ~1200 - 1600
//
// Calibration procedure:
// 1. Hold probe in open air and observe raw ADC -> record as SOIL_DRY_ADC.
// 2. Submerge probe to indicator line in water -> record as SOIL_WET_ADC.
// ----------------------------------------------------------------------------
const int SOIL_DRY_ADC = 3500;  // Raw ADC in dry air / 0% moisture
const int SOIL_WET_ADC = 1500;  // Raw ADC in saturated water / 100% moisture

// Set to true to print raw ADC readings alongside JSON for calibration
const bool DEBUG_RAW_ADC = false;

// Initialize DHT instance
DHT dht(DHT_PIN, DHT_TYPE);

void setup() {
  Serial.begin(115200);
  
  // Wait for serial monitor to stabilize
  delay(1000);

  // Initialize DHT sensor
  dht.begin();

  // Configure ADC attenuation (11dB attenuation allows full 0 - 3.3V measurement range)
  analogSetAttenuation(ADC_11db);

  delay(1500);
}

void loop() {
  // 1. Read DHT22 Digital Temperature (°C) and Humidity (% RH)
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  // 2. Read Analog Soil Moisture on GPIO 34
  // Average multiple ADC samples for noise reduction
  long adc_sum = 0;
  const int NUM_SAMPLES = 8;
  for (int i = 0; i < NUM_SAMPLES; i++) {
    adc_sum += analogRead(SOIL_PIN);
    delay(10);
  }
  int raw_soil = adc_sum / NUM_SAMPLES;

  // 3. Map raw ADC to soil moisture percentage (0.0% to 100.0%)
  // Invert mapping: higher ADC = dryer, lower ADC = wetter
  float soil_moisture = 0.0f;
  if (SOIL_DRY_ADC != SOIL_WET_ADC) {
    soil_moisture = ((float)(raw_soil - SOIL_DRY_ADC) / (float)(SOIL_WET_ADC - SOIL_DRY_ADC)) * 100.0f;
  }
  // Clamp between 0.0% and 100.0%
  if (soil_moisture < 0.0f) soil_moisture = 0.0f;
  if (soil_moisture > 100.0f) soil_moisture = 100.0f;

  // 4. Debug output if enabled
  if (DEBUG_RAW_ADC) {
    Serial.print("// RAW_ADC: ");
    Serial.print(raw_soil);
    Serial.print(" -> Soil%: ");
    Serial.println(soil_moisture, 1);
  }

  // 5. Transmit formatted JSON packet over Serial
  if (isnan(temperature) || isnan(humidity)) {
    // DHT22 read failure: transmit valid soil moisture, but flag DHT error honestly
    Serial.print("{\"error\":\"DHT22_read_failed\",\"soil_moisture\":");
    Serial.print(soil_moisture, 1);
    Serial.println("}");
  } else {
    // Normal validated packet
    Serial.print("{\"temperature\":");
    Serial.print(temperature, 1);
    Serial.print(",\"humidity\":");
    Serial.print(humidity, 1);
    Serial.print(",\"soil_moisture\":");
    Serial.print(soil_moisture, 1);
    Serial.println("}");
  }

  // DHT22 requires at least 2.0 seconds between reads
  delay(2000);
}
