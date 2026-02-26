import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import joblib
from streamlit_drawable_canvas import st_canvas
from scipy import ndimage

st.set_page_config(
    page_title="AI som tolkar handskrivna siffror",
    page_icon="🖋️",
    layout="wide"
)

# DESIGN, Färgval, jag vill att knapparna har en skugga så det ser ut som att de svävar.
st.markdown(
    """
    <style>
    /* 1) Basen för hela appen: bakgrund + standard textfärg */
    .stApp {
        background:
            radial-gradient(circle at 20% 15%, rgba(140, 90, 255, 0.25), transparent 45%),
            radial-gradient(circle at 75% 25%, rgba(80, 210, 255, 0.12), transparent 45%),
            linear-gradient(180deg, #05040a 0%, #0b0720 55%, #05040a 100%);
        color: #ECEBFF;
    }

    /* 2) Tvingar rubriker och text att bli ljusa (så de syns på mörk bakgrund) */
    h1, h2, h3, p, div, span, label {
        color: #ECEBFF !important;
    }

    /* 3) Streamlit-knappar: rundade, gradient och "svävande" skugga */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, rgba(140,90,255,0.95), rgba(70,30,180,0.95));
        color: white;
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 0.70rem 1rem;
        box-shadow: 0 12px 26px rgba(0,0,0,0.42);
        font-weight: 700;
        transition: transform 0.06s ease-in-out;
    }

    /* Hover: knappen lyfter lite */
    div.stButton > button:hover {
        transform: translateY(-2px);
    }

    /* Active: knappen trycks ner */
    div.stButton > button:active {
        transform: translateY(0px);
        box-shadow: 0 8px 18px rgba(0,0,0,0.42);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("AI som tolkar handskrivna siffror")
st.caption("Maskininlärningsmodell tränad på MNIST - utvecklad av Elin Molvig")
st.write("Rita en siffra 0 till 9. Tryck på prediktion för att se vad modellen gissar. Välj sedan rätt eller fel så hjälper du modellen att bli bättre.")

MODEL_PATH = r"C:/Users/elin-/OneDrive/Documents/GitRep/Kunskapskontroll 2/mnist_svc_final.pkl"
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def preprocess_canvas_image(img_rgba: np.ndarray) -> np.ndarray:
    """
    Tar RGBA bilddata från canvas och gör om till en input som liknar MNIST.

    Steg vi gör:
    1. Lägg canvas på vit bakgrund (för att ta bort transparens)
    2. Konvertera till gråskala
    3. Invertera så att "bläck" blir vitt på svart, som MNIST
    4. Beskär runt det som är ritat så siffran hamnar centralt
    5. Lägg padding så siffran inte klipps
    6. Skala om till 28x28 pixlar (MNIST storlek)
    7. Flatten till en radvektor (1, 784) som SVC modellen kan ta emot

    Returnerar:
    En numpy array med shape (1, 784) med pixelvärden 0 till 255.
    """
    # Skapa en PIL ("Python Imaging Library") bild från numpy arrayen
    # convert("RGBA") säkerställer att vi har en förutsägbar format
    pil = Image.fromarray(img_rgba).convert("RGBA")

    # Canvas använder ofta transparens.
    # Vi lägger därför ritningen på en vit bakgrund så den blir en "vanlig" bild.
    white_bg = Image.new("RGBA", pil.size, (255, 255, 255, 255))
    convert_grey = Image.alpha_composite(white_bg, pil).convert("L")  # "L" betyder gråskala

    # I MNIST är bakgrunden svart och siffran vit.
    # Vår canvas är tvärtom: svart penna på vit bakgrund.
    # Därför inverterar vi bilden.
    inverted = ImageOps.invert(convert_grey)

    # Konvertera till numpy array för att kunna analysera pixlarna
    arr = np.array(inverted)

    # Hitta var det finns "bläck"
    ys, xs = np.where(arr > 20)

    if len(ys) == 0:
        raise ValueError("Du har inte ritat något.")    

    # Bounding box: min och max koordinater för ritningen
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    # Beskär bilden runt det ritade området
    cropped = inverted.crop((x0, y0, x1 + 1, y1 + 1))

    # Skala så största dimensionen blir 20 px (MNIST-standard)
    w, h = cropped.size

    if w > h:
        new_w = 20
        new_h = max(1, int(round(h * (20 / w))))
    else:
        new_h = 20
        new_w = max(1, int(round(w * (20 / h))))

    resized = cropped.resize((new_w, new_h), Image.Resampling.NEAREST)

    # Skapa tom 28x28 svart canvas
    canvas = Image.new("L", (28, 28), 0)

     # Centrera resized i canvas med bounding box först
    left = (28 - new_w) // 2
    top = (28 - new_h) // 2
    canvas.paste(resized, (left, top))

    img = np.array(canvas).astype(np.float32)

    # Kontroll: för mycket bläck
    # Vi räknar hur stor andel av pixlarna som innehåller bläck över en tröskel
    ink_threshold = 30  # vad som räknas som bläck
    ink_pixels = np.sum(img > ink_threshold)
    total_pixels = img.size
    ink_ratio = ink_pixels / total_pixels

    # Om användaren har fyllt för stor del av rutan så är det troligen kladd
    if ink_ratio > 0.15:
        raise ValueError("För mycket kladd. Rita en tydligare siffra.")

    # Tyngdpunkt (center of mass)
    cy, cx = ndimage.center_of_mass(img)
    if np.isnan(cx) or np.isnan(cy):
        raise ValueError("Du har inte ritat något.")

    # Flytta så tyngdpunkten hamnar i mitten (14,14)
    shift_x = int(round(14 - cx))
    shift_y = int(round(14 - cy))

    # order=0 för att behålla "kantig" NEAREST-känsla
    img = ndimage.shift(img, shift=(shift_y, shift_x), order=0, mode="constant", cval=0.0)

    x = img.reshape(1, -1)

    return x

model = load_model()

# Session state för pott och senaste prediktion. Det är en räknare som sparar resultat. Vanligtvis så kör python skriptet om och nollar "potten" efter varje uppdatering.
if "total" not in st.session_state:
    st.session_state.total = 0
if "correct" not in st.session_state:
    st.session_state.correct = 0
if "wrong" not in st.session_state:
    st.session_state.wrong = 0

if "last_pred" not in st.session_state:
    st.session_state.last_pred = None
if "last_prob" not in st.session_state:
    st.session_state.last_prob = None
if "last_probs" not in st.session_state:
    st.session_state.last_probs = None
if "last_x" not in st.session_state:
    st.session_state.last_x = None
if "awaiting_feedback" not in st.session_state:
    st.session_state.awaiting_feedback = False

# Layout på kolumner
left_col, right_col = st.columns([3, 2], vertical_alignment="top")

with left_col: #Detta är vår canvas, 14 på penselbredd är bäst för MNIST bilderna
    st.subheader("Rita här")
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=14,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=360,
        width=360,
        drawing_mode="freedraw",
        key="canvas"
    )

    st.caption("Skriv en siffra")

    if st.session_state.last_x is not None:
        preview = st.session_state.last_x.reshape(28, 28).astype(np.uint8)
        st.image(preview, caption="Efter preprocessing 28x28", width=170)

with right_col:
    st.subheader("Träffsäkerhet på testade siffror")
    total = st.session_state.total
    correct = st.session_state.correct
    wrong = st.session_state.wrong
    acc = (correct / total) if total > 0 else 0.0

    true, false = st.columns(2)
    true.metric("Rätt", correct)
    false.metric("Fel", wrong)
    st.metric("Träffsäkerhet", f"{acc*100:.1f}%")
    st.divider()

    st.subheader("Prediktion")
    do_predict = st.button("Prediktera", use_container_width=True)

    if do_predict:
        if canvas_result.image_data is None:
            st.warning("Rita en siffra först.")
        else:
            try:
                x = preprocess_canvas_image(canvas_result.image_data)
            except ValueError as e:
                st.warning(str(e))
                st.stop()

            pred = int(model.predict(x)[0])

            probs = None
            best_prob = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(x)[0]
                best_prob = float(np.max(probs))

            st.session_state.last_pred = pred
            st.session_state.last_prob = best_prob
            st.session_state.last_probs = probs
            st.session_state.last_x = x
            st.session_state.awaiting_feedback = True

            st.rerun()

    if st.session_state.last_pred is not None:
        pred = st.session_state.last_pred
        best_prob = st.session_state.last_prob
        probs = st.session_state.last_probs

        st.write(f"Modellen gissar: **{pred}**")

        if best_prob is not None:
            st.write(f"Säkerhet: **{best_prob*100:.1f}%**")
            if best_prob < 0.45:
                st.info("Jag är osäker. Rita igen och lite tydligare, gärna större och mer centrerat.")

        if probs is not None:
            st.caption("Topp 3 sannolikheter")
            top3 = np.argsort(probs)[-3:][::-1]
            for i in top3:
                st.write(f"{int(i)}: {float(probs[i])*100:.1f}%")

        st.divider()

        if st.session_state.awaiting_feedback:
            st.subheader("Var gissningen rätt?")
            c1, c2 = st.columns(2)

            with c1:
                if st.button("✅ Rätt", use_container_width=True):
                    st.session_state.total += 1
                    st.session_state.correct += 1
                    st.session_state.awaiting_feedback = False
                    st.success("Sparat som rätt.")
                    st.rerun()

            with c2:
                if st.button("❌ Fel", use_container_width=True):
                    st.session_state.total += 1
                    st.session_state.wrong += 1
                    st.session_state.awaiting_feedback = False
                    st.success("Sparat som fel.")
                    st.rerun()





