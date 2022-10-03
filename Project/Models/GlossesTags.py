import os

import numpy as np
from flair.data import Sentence
from flair.models import SequenceTagger

DATA_PATH = os.path.join("Data", "Phoenix14")

questions_words = ["wo", "woher", "wohin", "wann", "was", "wer", "wie", "warum", "wenn", "etwas"]
# pronouns_words = ["ich", "du", "sei", "er", "sie", "IHR"]
months = ['oktober', 'november', 'juli', 'februar', 'dezember', 'januar', 'mai', 'september', 'april', 'august',
          'maerz', 'juni']
# season = ['herbst', 'sommer', 'winter', 'fruehling']
days = ['montag', 'dienstag', 'mittwoch', 'donnerstag', 'freitag', 'samstag', 'sonntag']
locations = ['england', 'schottland', 'bremen', 'nordpol', 'brandenburg', 'allgaeu', 'polen', 'eifel', 'island',
             'schweden', 'bodensee', 'muenster', 'nordpol', 'rumaenien', 'ungarn', 'weser', 'deutschland', 'schottland',
             'amerika']
numbers = ['zeh', 'fuenf', 'fuenfzehn', 'zwoelf', 'sechszehn', 'hundert', 'fuenfzig', 'dreissig', 'siebte', 'erste',
           'zweite', 'sechste', 'zehnte', 'dritte', 'sechshundert', 'fuenfhundert', 'dreihundert', 'zwoelfte', 'elfte',
           'elf', 'neunzehnte', 'sechzig', 'vierte', 'achte', 'fuenfte', 'neunte', 'zuerst', 'erst']

TAGS_LIST = ['NULL', 'LTR', 'QST', 'MNT', 'DAY', 'LOC', 'NUM']
SAVED_MODEL_PATH = "Data/models"


def get_glosses_tags(gloss_vocab):
    path = f"{SAVED_MODEL_PATH}/glosses_tag.npy"
    # if os.path.exists(path):
    #     return np.load(path, allow_pickle=True).item()

    glosses_idx_list = gloss_vocab.get_itos()

    tagger_ner = SequenceTagger.load("flair/ner-german-large")
    tagger_pos = SequenceTagger.load("flair/upos-multi")

    gloss_tag = {}
    for i in range(len(glosses_idx_list)):
        gloss_name = glosses_idx_list[i]
        if gloss_name in gloss_vocab.specials_tokens:
            gloss_tag[i] = 'NULL'
        if len(gloss_name) == 1 or gloss_name == 'SCH' or gloss_name == "NN" or gloss_name == "MM":
            gloss_tag[i] = 'LTR'
            continue
        if gloss_name.lower() in questions_words:
            gloss_tag[i] = 'QST'
            continue
        if gloss_name.upper() in months:
            gloss_tag[i] = 'MNT'
            continue
        if gloss_name.lower() in days:
            gloss_tag[i] = 'DAY'
            continue
        if gloss_name.upper() in locations:
            gloss_tag[i] = 'LOC'
            continue
        if gloss_name.upper() in numbers:
            gloss_tag[i] = 'NUM'
            continue
        tmp = gloss_name.split("-")
        if len(tmp) == 2 and (tmp[0].lower() == "neg" or tmp[0].lower() == "nicht"):
            gloss = Sentence(tmp[1].lower())
        else:
            gloss = Sentence(gloss_name.lower())
        tagger_ner.predict(gloss)
        tagger_pos.predict(gloss)
        if 'ner' in gloss.annotation_layers.keys() and gloss.annotation_layers['ner'][0].value == 'LOC':
            gloss_tag[i] = 'LOC'
        else:
            if gloss.annotation_layers['upos'][
                0].score > 0.7:  # and gloss.annotation_layers['upos'][0].value in TAGS_LIST:
                gloss_tag[i] = gloss.annotation_layers['upos'][0].value
            else:
                gloss_tag[i] = "NULL"

    counter = {}
    for k, v in gloss_tag.items():
        if v not in counter.keys():
            counter[v] = []
        counter[v].append(k)

    np.save(path, gloss_tag)
    return gloss_tag


def get_ner_glosses_tags(gloss_vocab):
    path = f"{SAVED_MODEL_PATH}/glosses_tag.npy"
    # if os.path.exists(path):
    #     return np.load(path, allow_pickle=True).item()

    glosses_idx_list = gloss_vocab.get_itos()

    tagger_ner = SequenceTagger.load("flair/ner-german-large")

    gloss_tag = {}
    for i in range(len(glosses_idx_list)):
        gloss_name = glosses_idx_list[i]
        if len(gloss_name) == 1 or gloss_name == 'SCH' or gloss_name == "NN" or gloss_name == "MM":
            gloss_tag[i] = 'LTR'
            continue
        if gloss_name.lower() in questions_words:
            gloss_tag[i] = 'QST'
            continue
        if gloss_name.upper() in months:
            gloss_tag[i] = 'MNT'
            continue
        if gloss_name.lower() in days:
            gloss_tag[i] = 'DAY'
            continue
        if gloss_name.upper() in locations:
            gloss_tag[i] = 'LOC'
            continue
        if gloss_name.upper() in numbers:
            gloss_tag[i] = 'NUM'
            continue
        gloss = Sentence(gloss_name.lower())
        tagger_ner.predict(gloss)
        if 'ner' in gloss.annotation_layers.keys() and gloss.annotation_layers['ner'][0].value == 'LOC':
            gloss_tag[i] = 'LOC'

    return gloss_tag


def manually(gloss_vocab):
    noun = ['TIEF', 'REGEN', 'REGION', 'SONNE', 'WOLKE', 'GRAD', 'SCHNEE', 'BISSCHEN', 'GEWITTER', 'WETTER', 'WIND',
            'SCHAUER', 'NEBEL', 'BERG', 'STURM', 'TEIL', 'FLUSS', 'MITTE', 'LAND', 'TEMPERATUR', 'KUESTE', 'FROST',
            'HIMMEL', 'VERLAUF', 'LUFT', 'ANFANG', 'STERN', 'ZUSCHAUER', 'MEER', 'DRUCK', 'ZEIGEN-BILDSCHIRM', 'ORT',
            'SEE', 'WALD', 'VORSICHT', 'ORKAN', 'UNWETTER', 'ZONE', 'WARNUNG', 'BODEN', 'DIENST', 'GEFAHR', 'BEDEUTET',
            'METER', 'REST', 'UNTERSCHIED', 'GRUND', 'EIS', 'EINFLUSS', 'BLITZ', 'BRAND', 'GRENZE', 'BEISPIEL',
            'MISCHUNG', 'HOEHE', 'TAL', 'BURG', 'HERZ', 'DONNER', 'HAGEL', 'GRAUPEL', 'VORAUSSAGE', 'UEBERSCHWEMMUNG',
            'VOGEL', 'AUFLOCKERUNG', 'ACHTUNG', 'LOCH', 'DAUER', 'RICHTUNG', 'STRASSE', 'LITER', 'WECHSEL', 'SOLL',
            'VORTEIL', 'WIRBEL', 'WASSER', 'BEREICH', 'OBER', 'RISIKO', 'SCHLUSS', 'VERKEHR', 'ZENTIMETER',
            'HOCHWASSER', 'INSEL', 'POSITION', 'PUNKT', 'STEIN', 'UHR', 'PROZENT', 'PROBLEM', 'CHANCE',
            'ENTSCHULDIGUNG', 'FELD', 'FOLGE', 'QUADRATMETER', 'AUTO', 'BOEE', 'GLUECK', 'HAUPT', 'KILOMETER',
            'NACH-HAUSE', 'NIESELREGEN', 'QUELL', 'ZUSAMMENSTOSS', 'HAUSE', 'ZOOM', 'ZEITSKALA', 'ZAHL', 'WIESE',
            'WEIBER', 'VIDEO', 'URLAUB', 'UEBERFLUTUNG', 'TIEFDRUCKZONE', 'THEMA', 'TAU', 'STROM', 'START', 'STAMM',
            'STADT', 'SPORT', 'SONNENUNTERGANG', 'SEITE', 'SCHRANK', 'SCHNEEVERWEHUNG', 'SCHLAGSAHNE', 'SCHIRM', 'SAND',
            'ROSE', 'QUADRAT', 'PULLOVER', 'PFLANZE', 'PFEIL', 'NUMMER', 'NIEDERUNG', 'NATUR', 'MUND', 'MOEGLICHKEIT',
            'MATSCH', 'MASCHINE', 'MARKT', 'LUECKE', 'LANDSCHAFT', 'LAERM', 'KURVE', 'KUCHEN', 'KRISE', 'KORB', 'KOMMA',
            'KLEINIGKEIT', 'KARTE', 'KANAL', 'KALENDER', 'HUT', 'HUND', 'HINDERNIS', 'HAVEN', 'GOTT', 'GIPFEL',
            'GESCHWINDIGKEIT', 'GARTEN', 'FRONT', 'FREIZEIT', 'FLUT', 'FEIER', 'FACH', 'ERNTE', 'DURCHSCHNITT', 'DUNST',
            'BROCKEN', 'BLATT', 'BETT', 'BERUF', 'BELAESTIGUNG', 'BEGINN', 'BEDINGUNGEN', 'AUSSICHT', 'ANHALT', 'ANGST',
            'MOEGLICHKEIT', 'ABSCHNITT', 'ABSCHIED', 'ZWEIFEL', 'ZUG', 'WIRTSCHAFT', 'WEIN', 'T-SHIRT', 'STOERUNG',
            'STAU', 'SKI', 'SCHLAF', 'SCHAU', 'REKORD', 'RAUM', 'RAND', 'PAUSE', 'MOND', 'MOMENT', 'KOERPER', 'GOLD',
            'GLATTEIS', 'FRAGEZEICHEN', 'FLOCKEN', 'ERZ', 'ERDRUTSCH', 'DRUCKFLAECHE', 'BLUETE', 'BAUM', 'BAUER',
            'AUSNAHME', 'ZUSAMMENHANG', 'VERGLEICH', 'TEXT', 'SITZ', 'SITUATION', 'PFINGSTEN', 'LEUTE', 'LAGE',
            'KUEHLER', 'INTERNET', 'WIRBELSTURM', 'ZENTRUM', 'ACHTZIG', 'BITTE', 'CHAOS', 'DAMEN', 'ENDE', 'FOEHN',
            'GEBIRGE', 'HAAR', 'HERREN', 'HESSEN']
    verb = ['SEIN', 'KOMMEN', 'KOENNEN', 'WEHEN', "AUSSEHEN", 'BLEIBEN', 'SEHEN', 'SCHNEIEN', 'VERSCHWINDEN', 'HABEN',
            'HABEN2', 'STEIGEN', 'SINKEN', 'AUFLOESEN', 'GEFRIEREN', 'WUENSCHEN', 'VERAENDERN', 'SCHEINEN', 'UMWANDELN',
            'BEGRUESSEN', 'INFORMIEREN', 'BEDEUTEN', 'AUFZIEHEN', 'RUEGEN', 'PASSEN', 'SAGEN', 'MITTEILEN', 'RECHNEN',
            'TROPFEN', 'BADEN', 'GIBT', 'MUESSEN', 'TAUEN', 'STROEMEN', 'VERRINGERN', 'MERKEN', 'ABSINKEN', 'ZIEHEN',
            'SCHADEN', 'SCHAFFEN', 'SCHAUEN', 'UMKEHREN', 'VERBREITEN', 'BRAUCHEN', 'GEWESEN', 'BEWEGEN', 'FUEHLEN',
            'NAEHERN', 'ANKOMMEN', 'AUFTAUCHEN', 'ABWECHSELN', 'AUSSEHEN', 'GEBEN', 'PASSIEREN', 'AUFPASSEN',
            'BEKANNTGEBEN', 'BLOCKIEREN', 'BRINGEN', 'DREHEN', 'FAHREN', 'ZUSAMMENTREFFEN', 'WIRKEN', 'WERDEN', 'WASCH',
            'WARTEN', 'WACHSEN', 'VORSTELLEN', 'VERTREIBEN', 'VERTEILEN', 'VERSUCHEN', 'VERDICHTEN', 'VERBINDEN',
            'UNTERNEHMEN', 'UMSTELLEN', 'TRINKEN', 'TANKEN', 'SUCHEN', 'STREIFEN', 'STEHEN', 'SPRIESSEN', 'SCHWITZEN',
            'SCHUETZEN', 'SCHMELZEN', 'SCHAETZEN', 'RUECKEN', 'REDUZIEREN', 'ORIENTIEREN', 'MITZIEHEN', 'MITNEHMEN',
            'MITEILEN', 'MITBEKOMMEN', 'MESSEN', 'MEINEN', 'LESEN', 'LEBEN', 'LAUFEN', 'KRATZEN', 'KAPUTTGEGANGEN',
            'LAUFE', 'HOLEN', 'HALTEN', 'GLITZERN', 'GLAUBEN', 'GENIESSEN', 'FLIESSEN', 'FEHLT', 'ERHOEHEN',
            'ENTHALTEN', 'EINSCHRAENKEN', 'DENKEN', 'BESPRECHEN', 'AUFTEILEN', 'AUFHOEREN', 'AUFHEITERN', 'AUFFUELLEN',
            'AUFEINANDERTREFFEN', 'AUFBLUEHEN', 'ANSCHAUEN', 'ANSAMMELN', 'AENDERN', 'ABKUEHLEN', 'ABFALLEN', 'WARTEN',
            'WUERZ', 'WOHNEN', 'WISSEN', 'VERLAUFEN', 'TUN', 'TRENNEN', 'STOCKEN', 'SPUEREN', 'SPAZIEREN', 'RODELN',
            'KLAPPEN', 'HOEREN', 'GEHT', 'GEHOERT', 'GEHEN', 'ERFAHREN', 'ENTWICKELN', 'EISEN', 'BERUHIGEN',
            'BEOBACHTEN', 'AUSHALTEN', 'AUFKOMMEN', 'AUFKLAREN', 'WIE-GEBLIEBEN', 'VORBEREITEN', 'VERSCHIEBEN',
            'VERMEIDEN', 'TAUGEN', 'STRAHLEN', 'MOEGEN', 'LIEGEN', 'HOFFEN', 'FALLEN', 'WAR', 'AUSRICHTEN',
            'AUSWAEHLEN', 'BEKOMMEN', 'ERWARTEN', 'HAFTEN']
    adverb = ['SCHNELL', 'LANGSAM', 'UNGEFAEHR', 'WAHRSCHEINLICH', 'GLATT', 'BESTIMMT', 'MAESSIG', 'MINUS', 'PLUS',
              'MEISTENS', 'BESONDERS', 'VIEL', 'JETZT', 'DICHT', 'RICHTIG', 'AKTUELL', 'ZU-ENDE', 'WEG', 'PLOETZLICH',
              'AUCH', 'HAUPTSAECHLICH', 'DABEI', 'NOCH', 'SCHON', 'UEBERWIEGEND', 'MAL', 'TEILWEISE', 'WIEDER',
              'DESHALB', 'DAZU', 'FRUEH', 'SPEZIELL', 'HIER', 'MANCHMAL', 'TATSAECHLICH', 'DARUM', 'VORAUS', 'KAUM',
              'OFT', 'ZUERST', 'VIELLEICHT', 'IM-MOMENT', 'UNTEN', 'NOCHEINMAL', 'GANZTAGS', 'INSGESAMT', 'DAZWISCHEN',
              'BISHER', 'FAST', 'TROTZDEM', 'VORHER', 'DANACH', 'ERSTMAL', 'OBEN', 'STELLENWEISE', 'LEIDER', 'ZURUECK',
              'AUF-JEDEN-FALL', 'AUTOMATISCH', 'IMMER', 'ERST', 'GENAU', 'KNAPP', 'DANN', 'WIEVIEL', 'VOR-ALLEM',
              'SOWIESO', 'SICHER', 'PUENKTLICH', 'MINDESTENS', 'MEHRMALS', 'IRGENDWO', 'IRGENDWANN', 'HERAB', 'EWIG',
              'EINZELN', 'EINFACH', 'EIGENTLICH', 'EBEN', 'DREIMAL', 'DRAUSSEN', 'DIESMAL', 'DEMNAECHST', 'DARUNTER',
              'DARAUF', 'DANEBEN', 'DAFUER', 'BERGAB', 'AUSEINANDER', 'ZUSAMMEN', 'STRENG', 'RUNTER', 'SEHR',
              'BEISEITE', 'AUSSERGEWOEHNLICH', 'BALD', 'DAUERND', 'ENDLICH', 'PAAR']
    adj = ['ALLE', 'KURZ', 'SELTEN', 'HEFTIG', 'KRAEFTIG', 'SCHWER', 'LEICHT', 'SPAETER', 'LANG', 'MEHR', 'KLAR',
           'STARK', 'GUT', 'MILD', 'SCHOEN', 'DEUTSCH', 'WARM', 'KALT', 'KOMMEND', 'TROCKEN', 'KUEHL', 'SCHWACH',
           'WECHSELHAFT', 'FREUNDLICH', 'REIF', 'AKTIV', 'WICHTIG', 'WEIT', 'WAHR', 'FRISCH', 'MAXIMAL', 'LIEB',
           'BEWOELKT', 'WEITER', 'RUHIG', 'NEU', 'NAECHSTE', 'BESSER', 'HEISS', 'FEUCHT', 'TRUEB', 'WENIGER',
           'TAGSUEBER', 'SCHWARZ', 'SAUER', 'NASS', 'ERSTE', 'ANGENEHM', 'WERT', 'VERSCHIEDEN', 'ZWEITE', 'DASSELBE',
           'SUPER', 'VIERTE', 'LOCKER', 'FUENFTE', 'LETZTE' 'SCHWUEL', 'ZEHNTE', 'DRITTE', 'NIEDER', 'DICK', 'FLACH',
           'NEUNTE', 'NORMAL', 'EIN-PAAR', 'GRAU', 'TYPISCH', 'BLAU', 'GROSS', 'SELBE', 'FLAECHENDECKEND', 'SCHLIMM',
           'SPAET', 'STABIL', 'BEDECKT', 'ARM', 'UNSICHER', 'UNGEMUETLICH', 'GELB', 'ALT', 'HARMLOS', 'ROT',
           'SCHWIERIG', 'ANDERS', 'GEMUETLICH', 'HART', 'KLEIN', 'ANGEMESSEN', 'DUENN', 'ELFTE', 'FERTIG', 'FREI',
           'GEWOHNT', 'MOEGLICH', 'WUNDERSCHOEN', 'WUNDERBAR', 'VORUEBERGEHEND', 'VOLL', 'VERWOEHNT', 'VERSPAETET',
           'VEREINZELT', 'VERANTWORTLICH', 'UNWAHRSCHEINLICH', 'UMSTAENDLICH', 'TROPISCH', 'SPAETESTEN', 'SCHULD',
           'NIEDRIG', 'NEUNZEHNTE', 'MERKWUERDIG', 'LICHT', 'LAHM', 'KONSTANT', 'KOMPLETT', 'INTERESSANT',
           'HERVORRAGEND', 'HELL', 'GRUEN', 'GETRENNT', 'FEST', 'EXTREM', 'ERSCHROCKEN', 'ERLEICHERT', 'ENTFERNT',
           'EMPFINDLICH', 'ECHT', 'DURCHEINANDER', 'DUMM', 'BUNT', 'BLUMEN', 'BETROFFEN', 'BESTE', 'BERGAUF', 'BEIDE',
           'ALLGEMEIN', 'ZUFRIEDEN', 'TOLL', 'SPITZE', 'SCHLECHTER', 'SCHLECHT', 'REIN', 'POSITIV', 'ORANGE', 'HEILIG',
           'GROB', 'GESAMT', 'GENUG', 'GEMISCHT', 'ENTSPANNT', 'BRAUN', 'HOEHER']
    prep = ['NEBEN', 'BIS', 'IN', 'NACH', 'ZWISCHEN', 'WEIL', 'AB', 'MIT', 'VOR', 'VON', 'UEBER', 'SEIT', 'NAH',
            'UNTER', 'DURCH', 'AEHNLICH', 'FUER', 'OHNE', 'BEI', 'UM', 'AUF', 'PRO', 'AN', 'AUS', 'ZU', 'INNERHALB']
    quest = ['WIE', 'WENN', 'WIE', 'WAS', 'WARUM', 'WO', 'WOHER', 'ETWAS', 'WER', 'WANN']
    pronoun = ['EUCH', 'ES', 'ICH', 'DU', 'DIESE', 'UNS', 'IHR', 'SIE', 'MEIN', 'WIR', 'EUCH', 'UNSER']
    loc = ['BAYERN', 'EUROPA', 'SKANDINAVIEN', 'SACHSEN', 'FRANKREICH', 'ENGLAND', 'BERLIN', 'DAENEMARK', 'KOELN',
           'RUSSLAND', 'ITALIEN', 'SCHLESWIG', 'HOLSTEIN', 'GRIECHENLAND', 'GROSSBRITANNIEN', 'SPANIEN',
           'NIEDERSACHSEN', 'NORWEGEN', 'RHEIN', 'BELGIEN', 'BRANDENBURG', 'NORDSEE', 'AACHEN', 'BRITANNIEN',
           'FINNLAND', 'FRANKFURT', 'HAMBURG', 'HOLLAND', 'IRLAND', 'NORDRHEIN-WESTFALEN', 'OESTERREICH', 'PFALZ',
           'WUERTTEMBERG', 'neg-GRAD', 'DRESDEN', 'ERFURT', 'MAINZ', 'MUENCHEN', 'RHEINLAND', 'RHEINLAND-PFALZ',
           'SAARLAND', 'TSCHECHIEN', 'neg-NORD', 'AFRIKA', 'ALPENRAND', 'ALPENTAL', 'BADEN-WUERTTEMBERG', 'BRUCKBERG',
           'DUESSELDORF', 'HANNOVER', 'KANADA', 'KOBLENZ', 'KROATIEN', 'LAUSITZ', 'LEIPZIG', 'MECKLENBURG',
           'MECKLENBURG-VORPOMMERN', 'NORDRHEIN', 'OSTBAYERN', 'PORTUGAL', 'ROSTOCK', 'RUHRGEBIET', 'SACKGASSE',
           'SLOWAKEI', 'STUTTGART', 'VORPOMMERN', 'DEUTSCHLAND', 'ALLGAEU', 'SCHOTTLAND', 'THUERINGEN', 'POLEN' 'EIFEL',
           'SYLT', 'ISLAND', 'SCHWEDEN', 'WESER', 'UNGARN', 'RUMAENIEN', 'ATLANTIK', 'AMERIKA', 'KIEL', 'TUERKEI',
           'BODENSEE', 'BREMEN', 'MUENSTER', 'NORDPOL', 'POMMERN']
    ltr = ['E', 'L', 'B', 'Z', 'M', 'R', 'F', 'H', 'I', 'S', 'A', 'U', 'J', 'T', 'V', 'W', 'C', 'D', 'P', 'SCH', 'Y',
           'K', 'G', 'N', 'O', 'MM', 'NN']
    time = ['MORGEN', 'NACHT', 'HEUTE', 'TAG', 'SONNTAG', 'FREITAG', 'SAMSTAG', 'DONNERSTAG', 'MITTWOCH', 'MONTAG',
            'DIENSTAG', 'MITTAG', 'WOCHENENDE', 'WOCHE', 'OKTOBER', 'NOVEMBER', 'SOMMER', 'JULI', 'WINTER', 'FEBRUAR',
            'DEZEMBER', 'APRIL', 'AUGUST', 'JUNI', 'NACHMITTAG', 'MAERZ', 'STUNDE''JANUAR', 'MAI', 'SEPTEMBER', 'ZEIT',
            'FRUEHLING', 'EIN-PAAR-TAGE', 'MORGENS', 'JAHR', 'VORMITTAG', 'GESTERN', 'AM-TAG', 'UEBERMORGEN', 'TAGE',
            'MONAT', 'WEIHNACHTEN', 'RUND-UM-DIE-UHR', 'JEDEN-TAG', 'MONATE', 'ABEND']
    direction = ['NORD', 'SUED', 'OST', 'WEST', 'NORDWEST', 'SUEDOST', 'NORDOST', 'SUEDWEST', 'OSTERN']
    num = ['ZWANZIG', 'FUENF', 'EINS', 'ZWEI', 'DREI', 'VIER', 'SIEBEN', 'ZEHN', 'ACHT', 'SECHS', 'NULL', 'DREISSIG',
           'VIERZEHN', 'NEUN', 'FUENFZEHN', 'ZWOELF', 'DREIZEHN', 'SIEBZEHN', 'ELF', 'SECHSZEHN', 'ACHTZEHN',
           'NEUNZEHN', 'SIEBTE', 'SECHSTE', 'ACHTE', 'HUNDERT', 'SECHSHUNDERT', 'FUENFHUNDERT', 'HAELFTE', 'SECHZIG',
           'NEUNZIG', 'SIEBZIG', 'VIERHUNDERT', 'VIERZIG', 'DREIHUNDERT', 'SIEBENHUNDERT', 'TAUSEND', 'ZWOELFTE',
           'ACHTHUNDERT', 'NEUNHUNDERT', 'FUENFZIG', 'HALB']
    conjunction = ['ABER', 'SONST', 'UND', 'WENN', 'ODER', 'SO', 'DOCH', 'ALS', 'OBWOHL', 'ALSO', 'OB', 'NUR']
    no = ['NEIN' 'KEIN', 'NICHT', 'neg', 'NICHTS']

    tags = {'NOUN': noun, 'VRB': verb, 'ADV': adverb, 'ADJ': adj, 'PREP': prep, 'QST': quest, 'PRN': pronoun,
            'LOC': loc, 'LTR': ltr, 'DIR': direction, 'TIME': time, 'NUM': num, 'CON': conjunction, 'NON': no}
    tag_gloss = {g: 'NONE' for g in gloss_vocab.get_itos()}
    for t, lst in tags.items():
        for g in tags[t]:
            if g in tag_gloss.keys():
                tag_gloss[g] = t
    return tag_gloss

