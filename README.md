# Interfață Vocală Modulară pentru Colaborare Om-Robot (Lucrare de Licență)

Sistem de asistență vocală pentru asamblare colaborativă om-robot, care combină:
- **LLM (OpenAI)** pentru înțelegerea comenzilor vocale și generarea răspunsurilor în limba română
- **RAG (Retrieval-Augmented Generation)** peste manualul de asamblare (PDF), folosind ChromaDB
- **Viziune artificială (YOLOv8 + Intel RealSense)** pentru detecția pieselor din scenă
- **Control robot (xArm) prin MQTT** pentru execuția pașilor de asamblare
- **Speech-to-Text / Text-to-Speech** pentru interacțiune vocală naturală
- **Interfață grafică (Tkinter)** cu feed video live și afișarea paginii relevante din manual

## Structură proiect

| Fișier | Rol |
|---|---|
| `licenta.py` | Punct de pornire al aplicației (thread dialog + TTS) |
| `dialog.py` | Bucla principală de conversație vocală |
| `agent_agent.py` | Logica agentului: interpretare intenții, apeluri LLM, RAG |
| `agent_intents.py` | Detectarea intențiilor din text (referințe la obiecte etc.) |
| `agent_state.py` | Starea curentă a scenei/focus-ului pe piese |
| `robot_bridge.py` | Comunicare cu robotul (protocol custom peste socket) |
| `varianta_finala_robot_MQTT.py` | Client MQTT + control xArm |
| `vision_realsense.py` | Captură video RealSense + detecție YOLOv8 |
| `speach_to_text.py` | Înregistrare audio + transcriere (Whisper) |
| `text_to_speach.py` | Sinteză vocală (gTTS) |
| `gui_interface_ajustat_final.py` | Interfața grafică principală |
| `populate_database.py` | Construiește baza vectorială ChromaDB din manualul PDF |
| `get_embedding_function.py` | Funcția de embeddings (OpenAI) |
| `app_config.py` | Configurări globale (MQTT, RAG, praguri, culori clase) |
| `utils_env.py` | Încarcă `.env` și inițializează clientul OpenAI |
| `test_*.py` | Scripturi de testare individuale pentru componente |

## Cerințe

- Python 3.10+
- Cont OpenAI cu API key
- Cameră Intel RealSense (pentru modulul de viziune)
- Robot xArm (opțional, pentru execuția fizică a pașilor)

Instalare dependențe (creează întâi `requirements.txt`, vezi mai jos):

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Configurare

1. Creează un fișier `.env` în rădăcina proiectului (NU se urcă pe GitHub):
   ```
   OPENAI_API_KEY=sk-proj-...
   ```
2. Verifică/ajustează parametrii din `app_config.py` (broker MQTT, topicuri, praguri RAG).
3. Rulează o singură dată, pentru a construi baza vectorială din manualul PDF:
   ```bash
   python populate_database.py
   ```

## Rulare

```bash
python licenta.py
```

sau, pentru interfața grafică:

```bash
python gui_interface_ajustat_final.py
```

## Notă

Fișierele `.pt` (greutăți model YOLO), directorul `chroma/` (bază vectorială generată) și fișierele audio temporare nu sunt incluse în acest repository (vezi `.gitignore`) — se regenerează local conform pașilor de mai sus.

## Context academic

Proiect de licență axat pe interacțiune vocală modulară bazată pe LLM-uri pentru colaborare om-robot, cu aplicare practică în asamblare industrială asistată.
