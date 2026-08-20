# agent_intents.py
import re


def _norm(user_text: str) -> str:
    t = user_text.lower().strip()
    t = t.replace('ş', 'ș').replace('Ş', 'Ș').replace('ţ', 'ț').replace('Ţ', 'Ț')
    t = re.sub(r"[^\w\săâîșț]", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t





def refers_to_current_object(user_text: str) -> bool:
    t = _norm(user_text)
    keywords = [
        "asta", "aceasta", "acesta", "această", "acest",
        "piesa", "piesă", "componenta", "componentă",
        "pe care o vezi", "care o vezi", "din camer", "de acolo",
        "cea mai apropiata", "cea mai apropiată", "cea apropiata", "cea apropiată"
    ]
    return any(k in t for k in keywords)


def intent_what_do_you_see(user_text: str) -> bool:
    t = _norm(user_text)
    return (
        ("ce vezi" in t) or
        ("ce piese" in t) or
        ("ce obiect" in t) or
        ("ce ai in fata" in t) or
        ("ce ai în față" in t)
    )


def intent_where_do_i_put_it(user_text: str) -> bool:
    t = _norm(user_text)
    return (
        ("unde o pun" in t) or
        ("unde o punem" in t) or
        ("unde se pune" in t) or
        ("unde merge" in t) or
        ("unde vine" in t)
    )


def intent_what_next(user_text: str) -> bool:
    t = _norm(user_text)
    return (
        ("ce urmeaza" in t) or
        ("ce urmează" in t) or
        ("pasul urmator" in t) or
        ("pasul următor" in t) or
        ("urmatorul pas" in t) or
        ("următorul pas" in t)
    )


def intent_what_is_it_for(user_text: str) -> bool:
    t = _norm(user_text)
    return (
        ("la ce ajuta" in t) or
        ("la ce ajută" in t) or
        ("la ce foloseste" in t) or
        ("la ce folosește" in t) or
        ("rol are" in t) or
        ("ce face" in t)
    )


def intent_doc_question(user_text: str) -> bool:
    t = _norm(user_text)
    keywords = [
        "ce contine", "ce conține", "echipament", "componente", "continut", "conținut",
        "manual", "instruc", "asamblare", "montaj", "pasi", "pași",
        "reguli", "setup", "pregatire", "pregătire",
        "pas", "etapa", "etapă", "verificari", "verificări",
        "probleme", "erori", "corectie", "corecție", "detalii", "explica", "explică"
    ]
    return any(k in t for k in keywords)


def intent_first_step(user_text: str) -> bool:
    t = _norm(user_text)
    keywords = [
        "primul pas",
        "care este primul pas",
        "care e primul pas",
        "cu ce incep",
        "cu ce încep",
        "cum incep asamblarea",
        "cum încep asamblarea",
        "inceputul asamblarii",
        "începutul asamblării"
    ]
    return any(k in t for k in keywords)


def intent_init_robot(user_text: str) -> bool:
    t = _norm(user_text)
    keywords = [
        "porneste robotul", "pornește robotul",
        "initializeaza robotul", "inițializează robotul",
        "pregateste robotul", "pregătește robotul",
        "porneste procesul", "pornește procesul",
        "start robot", "start proces",
        "initializare", "inițializare",
    ]
    return any(k in t for k in keywords)


def intent_details(user_text: str) -> bool:
    t = _norm(user_text)
    keywords = [
        "1",
        "detalii",
        "mai multe detalii",
        "da mi detalii",
        "dă mi detalii",
        "spune mi mai multe",
        "spune-mi mai multe",
        "explica",
        "explică",
        "vreau detalii",
        "mai multe informatii",
        "mai multe informații"
    ]
    return any(t == k or (len(k) > 1 and k in t) for k in keywords)


def intent_verifications(user_text: str) -> bool:
    t = _norm(user_text)
    keywords = [
        "2",
        "ce verific",
        "ce trebuie sa verific",
        "ce trebuie să verific",
        "cum verific",
        "verificari",
        "verificări",
        "validare",
        "ce trebuie verificat",
        "cum stiu ca e corect",
        "cum știu că e corect"
    ]
    return any(t == k or (len(k) > 1 and k in t) for k in keywords)


def intent_possible_errors(user_text: str) -> bool:
    t = _norm(user_text)
    keywords = [
        "3",
        "ce probleme pot aparea",
        "ce probleme pot apărea",
        "probleme posibile",
        "erori posibile",
        "ce poate merge prost",
        "ce se poate intampla gresit",
        "ce se poate întâmpla greșit",
        "ce greseli pot aparea",
        "ce greșeli pot apărea"
    ]
    return any(t == k or (len(k) > 1 and k in t) for k in keywords)


def intent_corrections(user_text: str) -> bool:
    t = _norm(user_text)
    keywords = [
        "4",
        "cum corectez",
        "ce fac daca e gresit",
        "ce fac dacă e greșit",
        "cum repar",
        "corectie",
        "corecție",
        "cum rezolv",
        "ce fac mai departe daca nu e bine"
    ]
    return any(t == k or (len(k) > 1 and k in t) for k in keywords)


def intent_continue(user_text: str) -> bool:
    t = _norm(user_text)
    keywords = [
        "continua", "continuă", "mai departe", "urmatorul",
        "următorul", "urmatorul pas", "următorul pas", "gata"
    ]
    return any(k in t for k in keywords)


def intent_pick_action(user_text: str) -> bool:
    t = _norm(user_text)
    keywords = [
        "ridica", "ridică",
        "ia", "preia",
        "apuca", "apucă",
        "prinde", "pick"
    ]
    return any(k in t for k in keywords)


def intent_place_action(user_text: str) -> bool:
    t = _norm(user_text)
    keywords = [
        "pune", "aseaza", "așază",
        "lasa", "lasă",
        "drop", "plaseaza", "plasează"
    ]
    return any(k in t for k in keywords)


def intent_confirm_yes(user_text: str) -> bool:
    t = _norm(user_text)
    keywords = [
        "da", "da executa", "da execută", "confirma", "confirmă",
        "ok", "bine", "executa", "execută", "sigur"
    ]
    return any(k in t for k in keywords)


def intent_confirm_no(user_text: str) -> bool:
    t = _norm(user_text)
    keywords = [
        "nu", "anuleaza", "anulează", "stop", "renunta", "renunță",
        "nu executa", "nu execută"
    ]
    return any(k in t for k in keywords)


def intent_yes_extra_info(user_text: str) -> bool:
    t = _norm(user_text)
    keywords = [
        "da", "da vreau", "vreau", "sigur", "ok", "bine",
        "spune", "da spune", "explica", "explică",
        "vreau informatii", "vreau informații",
        "1", "2", "3", "4"
    ]
    return any(t == k or (len(k) > 1 and k in t) for k in keywords)


def intent_no_extra_info(user_text: str) -> bool:
    t = _norm(user_text)
    keywords = [
        "nu", "nu vreau", "continua", "continuă", "mai departe",
        "urmatorul", "următorul", "gata"
    ]
    return any(k in t for k in keywords)