"""
Trains a data model with MOTOR-TALK forum posts (from the “data” directory) and classifies some example texts.
"""

import time

from utils import reduce_logging_output, load_dataset, create_and_train, print_analysis

# ------
# Config
# ------

data_base_path = "data"
text_embedding_module = "https://tfhub.dev/google/nnlm-de-dim128/1"
training_steps = 1000

# ---------
# Load Data
# ---------

print("*** LOADING ***")

start_time = time.time()

reduce_logging_output()

training_data = load_dataset(data_base_path)

print("dataset loading DONE:")
print(training_data.head())

# ---------------
# Train Estimator
# ---------------

print("*** TRAINING ***")
estimator = create_and_train(training_data, text_embedding_module, training_steps)
print("estimator training DONE")

# ---------------------------------
# Predict polarity of a single text
# ---------------------------------

mt_positive = "Guten Tag, Nachdem in diesem, sowie in anderen Foren sehr häufig nachgefragt wird, ob ein " \
              "Angebot auf einer der bekannten Automobilhandelsplattformen der Realität entspricht und leider " \
              "auch mittlerweile Betrugsopfer bei MT vorhanden sind, möchte Ich diese Problematik hier " \
              "einmal zusammenfassen. Vermutlich werde ich nicht in der Lage sein diesen Beitrag erfüllend " \
              "zu schreiben und werde sicherlich die ein oder andere Masche nicht aufzählen, die da noch " \
              "existieren mag. Doch fangen wir mal an. In der Regel findet man solche Angebote auf den " \
              "am stärksten frequentierten Automobil-Handelsseiten wie z.B. Autoscout24.de oder " \
              "mobile.de, doch auch auf Ebay.de wurden schon entsprechende Angebote gesehen. Ein " \
              "solches Angebot sieht meist wie folgt aus. Ein allgemein auf dem deutschen bzw. europäischen " \
              "Markt stärker nachgefragtes Fahrzeug wird deutlich unterhalb seines üblichen Marktpreises " \
              "zum Verkauf ausgeschrieben. Hierbei fallen mir folgende Fahrzeuge ein, bei welchen Ich " \
              "bereits solche Fakeangebote gesehen habe, zum einen handelt es sich um die volle Bandbreite " \
              "an Sportwagen vom Alfa Romeo Spider bis hin zum japanischen Tuningkultobjekt Toyota " \
              "Supra. Aber auch solide Fahrzeuge der gehobenen Mittelklasse, wie aktuelle E-Klassen oder " \
              "5er BMW tauchen hierbei häufig auf. Die vom Verkäufer aufgerufenen Verkaufspreise " \
              "erscheinen vom Standpunkt des normalen Marktpreises her als lächerlich. Oftmals handelt es " \
              "sich hierbei um Preise, die sogar deutlich unter denen für verunfallte Fahrzeuge zum " \
              "Ausschlachten liegen. So tauchten bereits 530er Diesel von BMW der aktuellen Baureihe " \
              "für unter 10.000 Euro in der Angebotspalette der Automobilbörsen auf. Sehr häufig fallen " \
              "diese Angebote dem Aufmerksamen Interessenten hier bereits auf. So haben die Fahrzeuge " \
              "oftmals keine europäischen Kennzeichen, oder es handelt sich sogar um Pressefotos des " \
              "Herstellers. Weiterhin ist in der Regel kein NAME und KEINE Adresse angegeben, oftmals " \
              "gibt es nur eine Angabe der Stadt (oftmals die wenigen International bekannten wie Hamburg, " \
              "München, Berlin, Köln). Eine angegebene Telefonnummer passt nicht zum Ort und entspricht " \
              "meist auch keiner deutschen Handynummernvorwahl. Schickt man dem Anbieter per Email einige " \
              "Fragen, erhält man in der Regel eine Antwortemail auf englisch. In dieser steht sinngemäß " \
              "immer das das Auto in BESTZUSTAND sei, keinerlei Kratzer oder Beulen habe und selbstredend " \
              "auch unfallfrei sei. Als Grund für den sehr niedrigen Verkaufspreis sind z.B. persönliche " \
              "Schicksale (Vater gestorben, Bruder verunfallt) oder spontane berufliche Veränderungen " \
              "(Neuer Beruf in Angola, Australien etc.pp.) und die Unmöglichkeit das Fahrzeug einzuführen " \
              "oder ähnliches (z.B. weil es Linkslenker sei etc.pp.). Das Fahrzeug, welches natürlich eine " \
              "deutsche Zulassung hat, steht aber in Spanien oder England oder oder oder (Warum wenn es ein " \
              "deutsches Auto ist ???). Sollte Interesse vorhanden sein, dann würde man das Auto zum " \
              "Interessenten bringen."
mt_negative = "was macht denn der newsthread hier im test forum?! *wunder* achja, zum zusammen führen " \
              "brauchst du diesen standart link http://www.motor-talk.de/showthread.php?threadid=XXXXXX " \
              "und bei XXXXX setzt du die zahl des post/threads ein, aber ohne das t davor!"
my_negative = "Ich kenne mich voll gut mit Autos aus. Mein Skoda Octavia ist super. Dein Auto ist total doof. " \
              "Willst Du was auf die Fresse? Dann beleidige mein Auto. Das ist der beste Weg, mich sehr " \
              "wütend zu machen. Aber wenn Du willst, dass ich Dir ein Küsschen gebe, dann musst Du nur " \
              "sagen, dass mein Auto total geil ist. Was geileres als meine Karre gibt's gar nicht. Oder? " \
              "Hey Leute, wer auch so einen schicken Skoda hat, der soll sich mal melden, dann können wir " \
              "ein bisschen fachsimpeln. Zum Beispiel was das beste Motoröl ist oder sowas. Obwohl, weiß " \
              "ich eigentlich auch nicht. Wenn ichs recht überlege habe ich eigentlich keine Ahnung von " \
              "Autos. OK Leute, Tschüssikowski!!!11!!"
my_positive = "Guten Morgen Hatte eigentlich nur die Startautomatik am Vergaser neu verkabelt, dann " \
              "stand der Bus einige Tage. Danach drehte der Anlasser nicht mehr auf Zündstellung 3. " \
              "dann wieder doch, wenn man den Schlüssel etwas festhielt. Nun aber garnicht mehr, mit " \
              "folgenden Phänomen: Direktes Kabel vom Erregeranschluss zum Pluspols lässt den Anlasser " \
              "drehen. Ein an das Kabel zum Erregeranschluss verlängertes und mit Prüflampe 12V Kabel " \
              "minus Pol der Batterie, lässt die Prüflampe leuchten. Also Anlasser geht überbrückt und " \
              "auf dem Kabel ist auch Spannung? Wieso geht das nicht? Woher nimmt der Anlasser Masse? " \
              "Gibt es überhaupt eine extra Masseverbindung? Zündanlasschalter wurde gewechselt und die " \
              "Leitungen überprüft. Alles OK. Kurzschliessen ohne Schlüssel hilft auch nicht."
# print prediction
print("*** ANALYZING ***")
print_analysis(mt_positive, estimator)
print_analysis(mt_negative, estimator)
print_analysis(my_negative, estimator)
print_analysis(my_positive, estimator)

print("Elapsed time: {0:.0f} sec".format(time.time() - start_time))
