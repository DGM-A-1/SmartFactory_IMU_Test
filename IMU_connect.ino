#include <Wire.h>
#include <WebSocketsServer.h>
#include <ESP8266WiFi.h>

#define TCA_ADDR        0x70        // TCA9548A I2C 주소
#define MPU_ADDR        0x68        // MPU6050/6500 I2C 주소
#define NUM_CHANNELS    8
#define READ_INTERVAL   100ul       // ms, 브로드캐스트 간격
#define GYRO_SF         131.0       // GYRO_CONFIG=0x00(±250 dps)
#define INT_PIN         D5          // INT 한 선으로 묶인 입력 핀(예: D5=GPIO14)

// Wi-Fi
const char* ssid     = "S23";
const char* password = "dgm2025!";

// WebSocket 서버
WebSocketsServer webSocket(81);

// ===== 캘리브레이션을 위한 Bias =====
struct Bias3D { double b[3];};

Bias3D accelBias[NUM_CHANNELS];
Bias3D gyroBias [NUM_CHANNELS];

// 센서별 상태
double rollArr[NUM_CHANNELS], pitchArr[NUM_CHANNELS], yawArr[NUM_CHANNELS];
double rollOffset[NUM_CHANNELS], pitchOffset[NUM_CHANNELS], yawOffset[NUM_CHANNELS];

// ★ 각속도(dps) 최신값을 송신용으로 저장
double xDelAng[NUM_CHANNELS], yDelAng[NUM_CHANNELS], zDelAng[NUM_CHANNELS];

unsigned long lastMicrosCh[NUM_CHANNELS]; // 채널별 마지막 업데이트(us)
unsigned long lastBroadcast = 0;          // 마지막 전송(ms)

volatile bool intFlag = false;            // ISR 플래그

// ===== Kalman =====
class Kalman {
public:
  void init(double Q_angle_, double Q_gyro_, double R_measure_) {
    Q_angle = Q_angle_; Q_gyro = Q_gyro_; R_measure = R_measure_;
    angle = bias = 0;
    P[0][0] = P[0][1] = P[1][0] = P[1][1] = 0;
  }
  double getKalman(double accAngle, double gyroRate, double dt){
    angle += dt * (gyroRate - bias);
    P[0][0] += dt * (dt * P[1][1] - P[0][1] - P[1][0] + Q_angle);
    P[0][1] -= dt * P[1][1];
    P[1][0] -= dt * P[1][1];
    P[1][1] += Q_gyro * dt;
    double S = P[0][0] + R_measure;
    double K0 = P[0][0] / S;
    double K1 = P[1][0] / S;
    double y = accAngle - angle;
    angle += K0 * y;
    bias  += K1 * y;
    P[0][0] -= K0 * P[0][0];
    P[0][1] -= K0 * P[0][1];
    P[1][0] -= K1 * P[0][0];
    P[1][1] -= K1 * P[0][1];
    return angle;
  }
private:
  double Q_angle, Q_gyro, R_measure;
  double angle, bias;
  double P[2][2];
};

Kalman kalmanRoll[NUM_CHANNELS], kalmanPitch[NUM_CHANNELS];

// ===== 유틸 =====
void tcaselect(uint8_t ch) {
  Wire.beginTransmission(TCA_ADDR);
  Wire.write(ch < NUM_CHANNELS ? (1 << ch) : 0);
  Wire.endTransmission();
}

void IRAM_ATTR onInt() { intFlag = true; }

void webSocketEvent(uint8_t num, WStype_t type, uint8_t *payload, size_t length) {
  if (type == WStype_CONNECTED) Serial.printf("Client %u connected\n", num);
  else if (type == WStype_DISCONNECTED) Serial.printf("Client %u disconnected\n", num);
}

static inline void i2cWrite8(uint8_t dev, uint8_t reg, uint8_t val){
  Wire.beginTransmission(dev);
  Wire.write(reg); Wire.write(val);
  Wire.endTransmission(true);
}

static inline uint8_t i2cRead8(uint8_t dev, uint8_t reg){
  Wire.beginTransmission(dev);
  Wire.write(reg); Wire.endTransmission(false);
  Wire.requestFrom(dev, (uint8_t)1, (uint8_t)true);
  return Wire.read();
}

// === IMU 제조 공정 과정에서 결정 되어 온 바이어스 값 ===
void loadBiases(){
  for (uint8_t ch=0; ch<NUM_CHANNELS; ch++){
    accelBias[ch].b[0] = 0.0; accelBias[ch].b[1] = 0.0; accelBias[ch].b[2] = 0.0;
    gyroBias [ch].b[0] = 0.0; gyroBias [ch].b[1] = 0.0; gyroBias [ch].b[2] = 0.0;
  }

  //ch 0
  accelBias[0].b[0] =  -0.03356009552471706; accelBias[0].b[1] = -0.0013182871246910643; accelBias[0].b[2] = 0.08306120686489021;
  gyroBias [0].b[0] = -0.16893129770992366;  gyroBias [0].b[1] =  -1.514236641221374;  gyroBias [0].b[2] = -0.123015267175572519;

  //ch1
  accelBias[1].b[0] =  -0.050283348137213774; accelBias[1].b[1] = 0.01280249287615498; accelBias[1].b[2] = 0.024672443444295755;
  gyroBias [1].b[0] = -0.06858778625954198;  gyroBias [1].b[1] =  -1.0372137404580153;  gyroBias [1].b[2] = -2.433740458015267;

  //ch2
  accelBias[2].b[0] =  -0.06804718778020215; accelBias[2].b[1] = 0.018603454633889346; accelBias[2].b[2] = -0.11266136830320561;
  gyroBias [2].b[0] = -3.0021374045801523;  gyroBias [2].b[1] = 0.08286259541984732;  gyroBias [2].b[2] = -0.29030534351145034;

  //ch3
  accelBias[3].b[0] =  -0.02358611242970543; accelBias[3].b[1] =-0.0011136034222059; accelBias[3].b[2] = -0.04979393095917217;
  gyroBias [3].b[0] = -11.437595419847325;  gyroBias [3].b[1] =  -0.23091603053435114;  gyroBias [3].b[2] = 1.7350381679389313;

  //ch4
  accelBias[4].b[0] =  -0.04844578741797334; accelBias[4].b[1] = 0.002230353740700298; accelBias[4].b[2] = 0.022136230998824336;
  gyroBias [4].b[0] = -4.2774427480916035;  gyroBias [4].b[1] = 0.04003816793893129;  gyroBias [4].b[2] = -1.020496183206107;

  //ch5
  accelBias[5].b[0] =  -0.07591590035190077; accelBias[5].b[1] = 0.029698712823580165; accelBias[5].b[2] = 0.03760570518406203;
  gyroBias [5].b[0] = -3.5393893129770992;  gyroBias [5].b[1] =  0.8189312977099238;  gyroBias [5].b[2] = -0.8307633587786259;

  //ch6
  accelBias[6].b[0] =  -0.020336408093985847; accelBias[6].b[1] = 0.004814316585803113; accelBias[6].b[2] = 0.04549539181956963;
  gyroBias [6].b[0] = -7.650458015267175;  gyroBias [6].b[1] =  0.2905343511450382;  gyroBias [6].b[2] = 0.5775190839694656;

  //ch7
  accelBias[7].b[0] =  0.04188791286834159; accelBias[7].b[1] = 0.019550670297169614; accelBias[7].b[2] = -0.04222257843535426;
  gyroBias [7].b[0] = 0.33175572519083973;  gyroBias [7].b[1] =  -1.396564885496183;  gyroBias [7].b[2] = -1.3395038167938933;
}

// ===== 채널 초기화 =====
void initChannel(uint8_t ch, int numSamples){
  tcaselect(ch);
  delayMicroseconds(300);

  // 전원/범위/필터/샘플레이트
  i2cWrite8(MPU_ADDR, 0x6B, 0x00); // PWR_MGMT_1: sleep 해제
  i2cWrite8(MPU_ADDR, 0x1B, 0x00); // GYRO_CONFIG: ±250 dps
  i2cWrite8(MPU_ADDR, 0x1C, 0x00); // ACCEL_CONFIG: ±2g
  i2cWrite8(MPU_ADDR, 0x1A, 0x03); // CONFIG: DLPF=3 (~44Hz)
  i2cWrite8(MPU_ADDR, 0x19, 0x09); // SMPLRT_DIV: 100 Hz

  // INT 설정: Active-Low + Open-Drain + Latch + RAW_DATA_RDY_EN
  i2cWrite8(MPU_ADDR, 0x37, 0xE0); // INT_PIN_CFG
  i2cWrite8(MPU_ADDR, 0x38, 0x01); // INT_ENABLE

  kalmanRoll[ch].init(0.001, 0.003, 0.03);
  kalmanPitch[ch].init(0.001, 0.003, 0.03);
  rollArr[ch] = pitchArr[ch] = yawArr[ch] = 0.0;
  yawOffset[ch] = 0.0;

  xDelAng[ch] = yDelAng[ch] = zDelAng[ch] = 0.0;

  lastMicrosCh[ch] = 0;

  // === 초기 자세 오프셋 ===
  double axSum=0, aySum=0, azSum=0;
  for (int i=0;i<numSamples;i++){
    Wire.beginTransmission(MPU_ADDR); Wire.write(0x3B); Wire.endTransmission(false);
    if (Wire.requestFrom(MPU_ADDR, 6, true) != 6) continue;
    int16_t ax = (Wire.read()<<8) | Wire.read();
    int16_t ay = (Wire.read()<<8) | Wire.read();
    int16_t az = (Wire.read()<<8) | Wire.read();
    double xg = ax/16384.0 - accelBias[ch].b[0];
    double yg = ay/16384.0 - accelBias[ch].b[1];
    double zg = az/16384.0 - accelBias[ch].b[2];
    axSum += xg; aySum += yg; azSum += zg;
    delay(2);
  }
  double avgX = axSum/numSamples, avgY = aySum/numSamples, avgZ = azSum/numSamples;
  rollOffset[ch]  = atan2(avgY,  sqrt(avgX*avgX + avgZ*avgZ)) * 180.0/M_PI;
  pitchOffset[ch] = atan2(-avgX, sqrt(avgY*avgY + avgZ*avgZ)) * 180.0/M_PI;

  tcaselect(NUM_CHANNELS); // 비활성
}

void setup(){
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(400000); // 400kHz

  loadBiases();

  // INT 핀 
  pinMode(INT_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(INT_PIN), onInt, FALLING);

  const int numSamples = 300; // 초기 자세 평균 샘플 수
  for (uint8_t ch=0; ch<NUM_CHANNELS; ch++) initChannel(ch, numSamples);

  // Wi-Fi / WebSocket
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println("\nWiFi connected! IP=" + WiFi.localIP().toString());
  webSocket.begin();
  webSocket.onEvent(webSocketEvent);

  // ★★★ 안정성 향상
  webSocket.enableHeartbeat(15000, 3000, 2); // 15s마다 ping, 3s 대기, 2회 실패시 종료
}

// ===== 한 채널 서비스 =====
bool serviceOneChannel(uint8_t ch){
  tcaselect(ch);
  delayMicroseconds(300);

  uint8_t st = i2cRead8(MPU_ADDR, 0x3A);
  if ((st & 0x01) == 0) { tcaselect(NUM_CHANNELS); return false; }

  // 14바이트 버스트 (Acc 6 + Temp 2 + Gyro 6)
  Wire.beginTransmission(MPU_ADDR); Wire.write(0x3B); Wire.endTransmission(false);
  if (Wire.requestFrom(MPU_ADDR, (uint8_t)14, (uint8_t)true) != 14){
    tcaselect(NUM_CHANNELS); return false;
  }

  int16_t ax = (Wire.read()<<8)|Wire.read();
  int16_t ay = (Wire.read()<<8)|Wire.read();
  int16_t az = (Wire.read()<<8)|Wire.read();
  Wire.read(); Wire.read(); // Temp skip
  int16_t gx = (Wire.read()<<8)|Wire.read();
  int16_t gy = (Wire.read()<<8)|Wire.read();
  int16_t gz = (Wire.read()<<8)|Wire.read();

  // 단위 변환
  double ax_g = ax/16384.0, ay_g = ay/16384.0, az_g = az/16384.0;
  double gx_d = gx/GYRO_SF,   gy_d = gy/GYRO_SF,   gz_d = gz/GYRO_SF;

  // bias 보정
  double AcX = ax_g - accelBias[ch].b[0];
  double AcY = ay_g - accelBias[ch].b[1];
  double AcZ = az_g - accelBias[ch].b[2];

  double X_RATE = gx_d - gyroBias[ch].b[0];
  double Y_RATE = gy_d - gyroBias[ch].b[1];
  double Z_RATE = gz_d - gyroBias[ch].b[2];

  // ★ 최신 각속도(dps)를 송신용 버퍼에 저장 (예시 코드의 X/Y/Z_DEL_ANG 의미)
  xDelAng[ch] = X_RATE;
  yDelAng[ch] = Y_RATE;
  zDelAng[ch] = Z_RATE;

  // dt
  unsigned long nowUs = micros();
  double dt = (lastMicrosCh[ch] == 0) ? 0.01 : (nowUs - lastMicrosCh[ch]) / 1e6;
  lastMicrosCh[ch] = nowUs;

  // 칼만/적분
  double accRoll  = atan2(AcY, sqrt(AcX*AcX + AcZ*AcZ)) * 180.0/M_PI;
  double accPitch = atan2(-AcX, sqrt(AcY*AcY + AcZ*AcZ)) * 180.0/M_PI;

  rollArr[ch]  = kalmanRoll[ch].getKalman(accRoll,  X_RATE, dt);
  pitchArr[ch] = kalmanPitch[ch].getKalman(accPitch, Y_RATE, dt);
  yawArr[ch]  += Z_RATE * dt;

  tcaselect(NUM_CHANNELS);

  // ★★★ 서비스 루프 굶기지 않기
  webSocket.loop();
  yield();

  return true;
}

void loop(){
  // ★★★ 항상 최우선으로 서비스
  webSocket.loop();

  // === 인터럽트 처리: "단발 패스"만 수행 (while 제거) ===
  if (intFlag || digitalRead(INT_PIN) == LOW){
    intFlag = false;

    bool any = false;
    for (uint8_t ch=0; ch<NUM_CHANNELS; ch++){
      if (serviceOneChannel(ch)) any = true;

      // 디버그: CH0만 가볍게 출력 (보정 적용 예시)
      if (ch==0){
        Serial.printf(
          "Ch%u | GX:%6.2f GY:%6.2f GZ:%6.2f | R:%6.2f P:%6.2f Y:%6.2f\n",
          ch, xDelAng[ch], yDelAng[ch], zDelAng[ch],
          rollArr[ch], pitchArr[ch], yawArr[ch]
        );
      }
    }
    (void)any; // 필요 시 활용
  }

  // === 브로드캐스트 ===
  unsigned long now = millis();
  if (now - lastBroadcast >= READ_INTERVAL){
    lastBroadcast = now;

    // 길어질 수 있으니 미리 버퍼 예약 (필드 추가로 여유 증가)
    String data;
    data.reserve(256 + NUM_CHANNELS * 128);

    data = "{\"sensors\":[";
    for (uint8_t ch=0; ch<NUM_CHANNELS; ch++){
      // 예시 코드와 동일하게 '보정 미적용' ROLL/PITCH/YAW 전송
      data += String("{\"id\":") + ch +
              ",\"X_DEL_ANG\":" + String(xDelAng[ch],2) +
              ",\"Y_DEL_ANG\":" + String(yDelAng[ch],2) +
              ",\"Z_DEL_ANG\":" + String(zDelAng[ch],2) +
              ",\"ROLL\":"      + String(rollArr[ch],2) +
              ",\"PITCH\":"     + String(pitchArr[ch],2) +
              ",\"YAW\":"       + String(yawArr[ch],2) +
              "}";
      if (ch < NUM_CHANNELS-1) data += ",";

      // ★★★ 조립 중에도 이벤트 서비스
      webSocket.loop();
      yield();
    }
    data += "]}";
    webSocket.broadcastTXT(data);
  }
}
