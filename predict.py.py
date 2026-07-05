import cv2
import numpy as np
import tensorflow as tf
import time
import pyttsx3
import threading
import serial

# ===================== SETTINGS =====================
MODEL_PATH    = "model.h5"
IMG_SIZE      = 64
CONFIDENCE_TH = 0.90
HOLD_TIME     = 1.5
SERIAL_PORT   = 'COM4'   # Windows: 'COM4', Linux/Mac: '/dev/ttyUSB0'
BAUD_RATE     = 9600
# ====================================================

# ── Arduino Connect ───────────────────────────────────────────────────────────
print("Connecting to Arduino...")
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"Arduino connected on {SERIAL_PORT}!")
except Exception as e:
    print(f"Arduino NOT connected: {e}")
    arduino = None

# ── Load Model ────────────────────────────────────────────────────────────────
print("Loading model...")
model   = tf.keras.models.load_model(MODEL_PATH)
classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
print("Model loaded!")

# ── TTS Engine ────────────────────────────────────────────────────────────────
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    def _speak():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

def send_to_arduino(data):
    """Send string to Arduino safely."""
    if arduino and arduino.is_open:
        try:
            arduino.write(data.encode())
            print(f"  → Arduino: '{data}'")
        except Exception as e:
            print(f"  Arduino send error: {e}")

# ── State Variables ───────────────────────────────────────────────────────────
word_list    = []
current_word = ""
last_letter  = ""
hold_start   = None
hold_letter  = ""

X1, Y1, X2, Y2 = 150, 80, 350, 280

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not found!")
    exit()

# ── Helpers ───────────────────────────────────────────────────────────────────
def draw_rounded_rect(img, pt1, pt2, color, thickness, radius=15):
    x1,y1 = pt1; x2,y2 = pt2
    cv2.line(img,(x1+radius,y1),(x2-radius,y1),color,thickness)
    cv2.line(img,(x1+radius,y2),(x2-radius,y2),color,thickness)
    cv2.line(img,(x1,y1+radius),(x1,y2-radius),color,thickness)
    cv2.line(img,(x2,y1+radius),(x2,y2-radius),color,thickness)
    cv2.ellipse(img,(x1+radius,y1+radius),(radius,radius),180,0,90,color,thickness)
    cv2.ellipse(img,(x2-radius,y1+radius),(radius,radius),270,0,90,color,thickness)
    cv2.ellipse(img,(x1+radius,y2-radius),(radius,radius), 90,0,90,color,thickness)
    cv2.ellipse(img,(x2-radius,y2-radius),(radius,radius),  0,0,90,color,thickness)

def draw_conf_bar(frame, conf, x=20, y=300, w=200, h=18):
    cv2.rectangle(frame,(x,y),(x+w,y+h),(40,40,40),-1)
    color = (0,220,80) if conf>0.90 else (0,180,255) if conf>0.70 else (0,80,255)
    cv2.rectangle(frame,(x,y),(x+int(w*conf),y+h),color,-1)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(180,180,180),1)
    cv2.putText(frame,f"Confidence: {conf*100:.1f}%",
                (x,y-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)

def draw_hold_ring(frame, progress, cx=250, cy=180):
    angle = int(360 * progress)
    cv2.ellipse(frame,(cx,cy),(30,30),-90,0,360,(60,60,60),4)
    cv2.ellipse(frame,(cx,cy),(30,30),-90,0,angle,(0,255,180),4)

# ── Main Loop ─────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("  ASL Sign Language Predictor + Arduino")
print("="*50)
print("  SPACE  → Space between words")
print("  BKSP   → Delete last letter")
print("  ENTER  → Speak sentence")
print("  W      → Send full word to Arduino")
print("  S      → Send full sentence to Arduino")
print("  C      → Clear all")
print("  ESC    → Exit")
print("="*50)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h_fr, w_fr = frame.shape[:2]

    # ── Predict ───────────────────────────────────────────────────────────────
    roi  = frame[Y1:Y2, X1:X2]
    img  = cv2.resize(roi, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
    img  = np.expand_dims(img, axis=0)
    pred = model.predict(img, verbose=0)
    idx  = int(np.argmax(pred))
    conf = float(np.max(pred))
    letter = classes[idx]

    # ── Hold Logic ────────────────────────────────────────────────────────────
    hold_progress = 0.0

    if conf >= CONFIDENCE_TH:
        if letter == hold_letter:
            elapsed       = time.time() - hold_start
            hold_progress = min(elapsed / HOLD_TIME, 1.0)
            if elapsed >= HOLD_TIME and letter != last_letter:
                current_word += letter
                last_letter   = letter
                hold_start    = time.time()
                print(f"  Added: {letter}  | Word: {current_word}")

                # ✅ Send detected letter to Arduino immediately
                send_to_arduino(letter)

        else:
            hold_letter = letter
            hold_start  = time.time()
            last_letter = ""
    else:
        hold_letter   = ""
        hold_progress = 0.0

    # ── UI ────────────────────────────────────────────────────────────────────
    panel = frame.copy()
    cv2.rectangle(panel,(0,0),(w_fr,340),(0,0,0),-1)
    cv2.addWeighted(panel,0.5,frame,0.5,0,frame)

    box_color = (0,255,80) if conf>=CONFIDENCE_TH else (0,140,255)
    draw_rounded_rect(frame,(X1,Y1),(X2,Y2),box_color,2)

    if conf >= CONFIDENCE_TH:
        cv2.putText(frame, letter,
                    (X2+20, Y1+90), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,255,80), 5)
        draw_hold_ring(frame, hold_progress, cx=X2+70, cy=Y2-30)
        if hold_progress >= 1.0:
            cv2.putText(frame,"+ ADDED!",
                        (X2+10, Y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,150),2)
    else:
        cv2.putText(frame, "?",
                    (X2+40, Y1+90), cv2.FONT_HERSHEY_SIMPLEX, 4,(60,60,255),5)

    draw_conf_bar(frame, conf, x=20, y=295)

    # Arduino status indicator
    ard_status = "Arduino: CONNECTED" if (arduino and arduino.is_open) else "Arduino: OFFLINE"
    ard_color  = (0,255,100) if (arduino and arduino.is_open) else (0,80,255)
    cv2.putText(frame, ard_status, (20,85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ard_color, 1)

    sentence = " ".join(word_list) + (" " if word_list else "") + current_word
    cv2.putText(frame,"Word Building:",
                (20,350), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(180,180,255),1)
    cv2.rectangle(frame,(15,358),(w_fr-15,398),(20,20,50),-1)
    cv2.rectangle(frame,(15,358),(w_fr-15,398),(100,100,200),1)
    display_sentence = sentence if len(sentence)<=30 else "..."+sentence[-27:]
    cv2.putText(frame, display_sentence,
                (22,388), cv2.FONT_HERSHEY_SIMPLEX, 0.85,(255,255,100),2)

    hints = "SPACE=word | BKSP=del | ENTER=speak | W=word→ard | S=sent→ard | C=clear | ESC=exit"
    cv2.putText(frame, hints,
                (10, h_fr-10), cv2.FONT_HERSHEY_SIMPLEX, 0.38,(150,150,150),1)

    cv2.putText(frame,"ASL Sign Predictor + Arduino",
                (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255,220,0),2)
    cv2.putText(frame,f"Hold {HOLD_TIME}s to add",
                (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(200,200,200),1)

    cv2.imshow("ASL Sign Language Predictor", frame)

    # ── Key Handling ──────────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == 27:       # ESC
        break

    elif key == 32:     # SPACE → save word
        if current_word:
            word_list.append(current_word)
            send_to_arduino(" ")        # send space to Arduino
            print(f"  Word saved: {current_word}")
            current_word = ""
            last_letter  = ""

    elif key == 8:      # BACKSPACE
        if current_word:
            current_word = current_word[:-1]
            last_letter  = ""
            print(f"  Deleted. Word: {current_word}")

    elif key == 13:     # ENTER → speak
        to_speak = (" ".join(word_list) + " " + current_word).strip()
        if to_speak:
            print(f"  Speaking: {to_speak}")
            speak(to_speak)

    elif key in (ord('w'), ord('W')):   # W → send current word to Arduino
        if current_word:
            send_to_arduino(current_word)

    elif key in (ord('s'), ord('S')):   # S → send full sentence to Arduino
        full = (" ".join(word_list) + " " + current_word).strip()
        if full:
            send_to_arduino(full)

    elif key in (ord('c'), ord('C')):   # C → clear
        word_list    = []
        current_word = ""
        last_letter  = ""
        print("  Cleared!")

# ── Cleanup ───────────────────────────────────────────────────────────────────
cap.release()
if arduino and arduino.is_open:
    arduino.close()
cv2.destroyAllWindows()
print("\nFinal:", " ".join(word_list), current_word)
