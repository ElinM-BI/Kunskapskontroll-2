🖋️ AI som tolkar handskrivna siffror

Detta projekt är en komplett maskininlärningspipeline där jag:
- Utforskar och utvärderar flera modeller
- Motiverar valet av slutmodell (SVC)
- Sparar den tränade modellen
- Implementerar en interaktiv webbapp i Streamlit

Projektet bygger på MNIST-datasetet och demonstrerar både modellutveckling och praktisk implementation.

📂 Projektstruktur
📊 1. ML-modellering

Denna del innehåller:

Dataförberedelse
Modelljämförelse
Hyperparameteroptimering med GridSearchCV

Utvärdering med:
Accuracy

Precision
Recall
F1-score
Confusion Matrix

Flera modeller analyseras och jämförs. Valet av Support Vector Classifier (SVC) motiveras utifrån prestanda, stabilitet och generaliseringsförmåga.

💾 2. Sparad modell

Den bästa modellen sparas med joblib.

🌐 3. Streamlit-app

En interaktiv webbapplikation där användaren kan:

- Rita en siffra (0–9)
- Göra en prediktion
- Se sannolikheter (topp 3)
- Få modellens säkerhetsnivå
- Markera om modellen hade rätt eller fel
- Följa löpande träffsäkerhet

Appen använder:

st.session_state för att hantera statistik

Bildpreprocessing för att konvertera canvas till 28x28-format

predict() för klassificering

predict_proba() för sannolikhetsbedömning

🧠 Modellval: Varför SVC?

Efter jämförelse mellan flera algoritmer visade SVC:

Hög och stabil accuracy
God balans mellan precision och recall

Stark prestanda på MNIST-datasetet

Macro-average användes vid beräkning av precision, recall och F1-score för att säkerställa balanserad prestanda över samtliga klasser.

📈 Modellprestanda

Exempel på uppnådda resultat:

Accuracy ≈ 97%

Precision ≈ 0.97

Recall ≈ 0.97

F1-score ≈ 0.97

Detaljerad per-klass-analys genomfördes med classification_report.

⚙️ Teknologier

Python
Scikit-learn
NumPy
Matplotlib
Joblib
Streamlit
