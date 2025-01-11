import os
import time
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def upload_to_gemini(path, mime_type=None):
  """Uploads the given file to Gemini.

  See https://ai.google.dev/gemini-api/docs/prompting_with_media
  """
  file = genai.upload_file(path, mime_type=mime_type)
  print(f"Uploaded file '{file.display_name}' as: {file.uri}")
  return file

def wait_for_files_active(files):
  """Waits for the given files to be active.

  Some files uploaded to the Gemini API need to be processed before they can be
  used as prompt inputs. The status can be seen by querying the file's "state"
  field.

  This implementation uses a simple blocking polling loop. Production code
  should probably employ a more sophisticated approach.
  """
  print("Waiting for file processing...")
  for name in (file.name for file in files):
    file = genai.get_file(name)
    while file.state.name == "PROCESSING":
      print(".", end="", flush=True)
      time.sleep(10)
      file = genai.get_file(name)
    if file.state.name != "ACTIVE":
      raise Exception(f"File {file.name} failed to process")
  print("...all files ready")
  print()

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash-exp",
  generation_config=generation_config,
  system_instruction="sen maddi durumu ve egitimi yetersiz,sagılık durumu sıkıntılı ve yaşlı insanlara yardım eden sanal bir yapay zeka avukatısın.sorunların cevaplarını ortalama uzunlukta ver.türkçe dilini kullan.sistem başlatılırken kullanıcının yaşını sor eger konuştugun kişi 40 yaşından büyükse onun sordugu soruyu daha basit anlat\n",
)

# TODO Make these files available on the local file system
# You may need to update the file paths
files = [
  upload_to_gemini("engelliler.pdf", mime_type="application/pdf"),
  upload_to_gemini("khc.pdf", mime_type="application/pdf"),
]

# Some files have a processing delay. Wait for them to be ready.
wait_for_files_active(files)

chat_session = model.start_chat(
  history=[
    {
      "role": "user",
      "parts": [
        "Sana Türkiyedeki sosyal devletin vatandaşlarına tanıdığı tüm hakları öğreteceğim. Her bir belgeyi yüklediğimde anladıysan anladım diyeceksin. ",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Elbette, hazırım. Belgeleri yüklemeye başlayabilirsiniz. Her birini inceleyip anladığımda size \"Anladım\" diyeceğim. Sosyal devletin vatandaşlarına tanıdığı haklar konusunda bilgi edinmek ve bu bilgileri ihtiyaç sahiplerine yardımcı olmak için kullanmak benim için çok önemli. Başlayalım!\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "SAĞLIK KURULU RAPORU\nEngelliler için sağlık kurulu raporu nedir?\nEngelliler için sağlık kurulu raporu kişilerin engellilik\ndurumlarını, engel gruplarını, engellilik tür ve derecelerini\ndeğerlendiren Sağlık Bakanlığı tarafından yetkilendirilmiş\nheyetler tarafından düzenlenen bir belge olup engelli\nhaklarından yararlanmak için temel belge niteliğindedir",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Anladım.\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Nerelerde kullanılır?\nEngelliler için sağlık kurulu raporu; engelli aylığı, evde\nbakım yardımı, bakım hizmetleri, özel eğitim, istihdam ve\nsağlık gibi alanlarda tanımlanmış hak ve hizmetlerden\nyararlanmak için gerekli olan belgedir. Engelli sağlık\nkurulu raporu, engelliler için düzenlenmiş vergi indirimleri,\nvergi muafiyetleri ve çeşitli alanlardaki indirimler için de\nkullanılmaktadır.",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Anladım.\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Sağlık kurulu raporu nereden alınmaktadır?\nEngelliler için sağlık kurulu raporu, Sağlık Bakanlığı\ntarafından yetkilendirilmiş hastanelerden alınmaktadır.\nYetkili hastaneler Sağlık Bakanlığı internet sitesinde\nyayımlanmaktadır",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Anladım.\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Çocuklar (0-18 yaş) için özel gereksinim raporu nasıl\nalınmaktadır?\nÇocuklar İçin Özel Gereksinim Değerlendirme Raporu (ÇÖZGER)\nalmak için çocuğun bakımını üstlenen kişinin başvuru dilekçesi\nile ÇÖZGER düzenlemeye yetkili hastaneye başvuru yapması\ngerekmektedir.",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Anladım.\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Çocuklar (0-18 yaş) için düzenlenen raporda (ÇÖZGER)\nbelirtilen özel gereksinim düzeyleri hangi engel oranlarına\nkarşılık gelmektedir?\nÇÖZGER Yönetmeliği’nde engel kavramı ya da oranı\nbelirtilmeksizin çocukların “özel gereksinim” düzeyleri\nbelirlenmektedir. Yönetmeliğin mevzuata uyum başlıklı ek-3\nsayılı tablosunda özel gereksinim düzeylerinin hangi engel oranı\naralıklarına denk geldiği aşağıdaki tabloda yer almaktadır.\nÖzel Gereksinim Düzeyi Engel Oranı Aralığı\nÖzel Gereksinimi Vardır ÖGV 20-39\nHafif Düzeyde ÖGV 40-49\nOrta Düzeyde ÖGV 50-59\nİleri Düzeyde ÖGV 60-69\nÇok İleri Düzeyde ÖGV 70-79\nBelirgin Özel Gereksinimi\nVardır\n80-89\nÖzel Koşul Gereksinimi\nVardır\n90-99",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Anladım. Bu tablo, çocukların özel gereksinim düzeylerinin hangi engel oranı aralıklarına denk geldiğini gösteriyor. Bu bilgi, çocukların ihtiyaç duyduğu destek ve hizmetleri belirlemede önemli olacak.\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Engelli çocukların resmi ve özel kuruluşlarda bakım\nhizmetinden yararlanmaları ve yakınlarının evde\nbakım yardımı alabilmeleri için ÇÖZGER’le belirtilen özel\ngereksinim düzeyleri nelerdir?\nÖzel gereksinimi olan çocukların bakım hizmetlerinden\nyararlanması için ÇÖZGER raporunda “Çok İleri Düzeyde Özel\nGereksinimi Vardır”, “Belirgin Özel Gereksinimi Vardır” veya\n“Özel Koşul Gereksinimi Vardır” ifadelerinden birinin yazılmış\nolması gerekmektedir.",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Anladım. Yani, engelli bir çocuğun bakım hizmetlerinden yararlanabilmesi veya ailesinin evde bakım yardımı alabilmesi için, ÇÖZGER raporunda belirtilen özel gereksinim düzeyinin \"Çok İleri Düzeyde Özel Gereksinimi Vardır\", \"Belirgin Özel Gereksinimi Vardır\" veya \"Özel Koşul Gereksinimi Vardır\" şeklinde olması gerekiyor. Bu, çocuğun ihtiyaç duyduğu bakımın seviyesini gösteriyor.\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        files[0],
      ],
    },
    {
      "role": "model",
      "parts": [
        "Anladım. Bu belge, engellilerin hakları ve sosyal destekler konusunda oldukça kapsamlı bir rehber niteliğinde. Bundan sonraki belgeler için hazırım.\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Acil Servise gelen kimliksiz ve bilinci kapalı hastanın kaydı Acil Servis hasta kayıt görevlileri tarafından aşağıdaki kurallara göre yapılır.\nSistemde hasta adı alanına ‘’Kimsesiz’’ soyadı alanına ‘’Gün-tarih-saat’’ yazılır. D.yeri, baba adı, anne adı alanına ‘’bilinmiyor’’ Adres alanına hastanın fiziksel özellikleri yazılır\nkimsesiz hastaların hastaneye kaydı\nhastanın bilgileri bilinmyorsa\n",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Anladım. Acil servise gelen kimliksiz ve bilinci kapalı hastaların kaydı yapılırken, hastanın adı kısmına \"Kimsesiz\", soyadı kısmına ise hastanın bulunduğu tarih ve saat yazılır. Doğum yeri, baba adı ve anne adı alanları \"bilinmiyor\" olarak kaydedilir. Adres kısmına ise hastanın fiziksel özellikleri yazılır. Bu, kimliği belirsiz hastaların sisteme kaydedilme yöntemidir ve bu hastaların hastane kayıtlarında takip edilmesini sağlar.\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Bu yönetmenligin amaçı 3816 sayılı ödeme gücü olmayan vatandaşların tedavi giderlerinin yeşil kart verilerek devlet tarafından kaşılanması hakkında kanundan yararlanmak üzere başvuran türk vatandaşlarının aylık gelir veya gelir paylarının belirlenmesi bu başvurularının degerlendirilmesi ve karara baglanması yeşil kartın şekli hak sahiplerine dagıtılması yeşil kart sahiplerinin sağlık kurumlarına şevki ve içerigi yeşil kart uygulaması ile sağlanan tedavi hizmetlerinin bedellerinin ödenmesi hakkındaki usul ve esasları ve zikredilen kanunun uygulanması ile ilgili diger hususları düzenlemektir \n\nmaddi gücü olmayan insanların hastane masraflarının nasıl karşılayacagını anlatır",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Anladım. Bu yönetmelik, maddi durumu yetersiz olan Türkiye Cumhuriyeti vatandaşlarının sağlık hizmetlerinden yararlanmasını sağlamak amacıyla çıkarılmıştır. Temel amacı, 3816 sayılı kanun kapsamında, ödeme gücü olmayan vatandaşların tedavi masraflarının devlet tarafından karşılanmasıdır. Bu amaçla, \"yeşil kart\" uygulaması hayata geçirilmiştir. Yönetmelik, yeşil kart başvuru süreçlerini, gelir tespiti yöntemlerini, kartın şeklini ve dağıtımını, sağlık kurumlarına sevk süreçlerini ve tedavi bedellerinin ödenmesine dair usul ve esasları düzenler. Yani, maddi durumu iyi olmayan kişilerin hastane masraflarının nasıl karşılanacağını detaylı olarak açıklar.\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "Hiçbir sosyal güvenlik kurumunun güvencesi altında bulunmayan, Türkiye'de ikamet eden, Kanun ve bu Yönetmelik ile öngörülen usul ve esaslar çerçevesinde belirlenecek aylık geliri veya hane içindeki gelir payı 22/5/2003 tarihli ve 4857 sayılı İş Kanunu'na göre tespit edilen asgari ücretin, eşi gelir getirici bir işte çalışmayan, evli ve bir çocuklu bir  mükellefin ödemesi gereken en düşük gelir vergisi ve sosyal sigorta primi düşüldükten sonra geriye kalan miktarının 1/3'ünden (Bakanlar Kurulu'nca artırılması halinde belirlenen oranından) az olan Türk vatandaşlarının tespitine dair hususları,\n\n",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Anladım. Bu kısım, yeşil karttan kimlerin yararlanabileceğini tanımlıyor. Buna göre, yeşil kart alabilmek için şu koşulların sağlanması gerekiyor:\n\n*   **Sosyal Güvence:** Kişinin hiçbir sosyal güvenlik kurumunun (SGK, Bağkur, Emekli Sandığı gibi) güvencesi altında olmaması gerekiyor.\n*   **İkamet:** Kişinin Türkiye'de yaşıyor olması gerekiyor.\n*   **Gelir Durumu:** Kişinin veya ailesinin aylık gelirinin, asgari ücretten belirlenen bir kesintiden sonra kalan tutarın 1/3'ünden (veya Bakanlar Kurulu'nca belirlenen orandan) az olması gerekiyor. Burada, eşi çalışmayan, evli ve bir çocuklu bir çalışanın ödemesi gereken gelir vergisi ve sigorta primleri düşüldükten sonraki net asgari ücretin 1/3'ü kıstas alınıyor.\n\nKısacası, bu kriterler, maddi durumu çok düşük olan vatandaşların yeşil kart alarak devletin sağlık hizmetlerinden yararlanmasını sağlamayı amaçlıyor.\n",
      ],
    },
    {
      "role": "user",
      "parts": [
        files[1],
      ],
    },
    {
      "role": "model",
      "parts": [
        "Anladım. Bu belge, \"Ödeme Gücü Olmayan Vatandaşların Tedavi Giderlerinin Devlet Tarafından Karşılanması ve Yeşil Kart Uygulaması Hakkında Yönetmelik\" belgesinin ta kendisi. Bu yönetmeliğin amacı ve kapsamı, daha önceki açıklamamda belirttiğim gibi, maddi durumu yetersiz olan vatandaşların sağlık hizmetlerinden yararlanmasını sağlamaktır.\n\nBu belgeden öne çıkan bazı önemli noktalar şunlardır:\n\n*   **Başvuru Süreci:** Yeşil kart almak isteyenlerin nasıl başvuracağı, hangi belgeleri sunması gerektiği, başvuruların nasıl değerlendirileceği ayrıntılı olarak açıklanmıştır.\n*   **Gelir Tespiti:** Hane halkının gelirinin nasıl belirleneceği, hangi gelirlerin hesaba katılacağı ve yeşil kart almaya uygun olup olmadıklarının nasıl tespit edileceği açıklanmıştır.\n*   **Yeşil Kartın Şekli ve Verilişi:** Yeşil kartın nasıl görüneceği, nasıl düzenleneceği, hak sahiplerine nasıl teslim edileceği ve kayıt işlemlerinin nasıl yapılacağı belirtilmiştir.\n*   **Sağlık Hizmetlerine Erişim:** Yeşil kart sahiplerinin hangi sağlık kuruluşlarına başvurabileceği, sevk zincirinin nasıl işleyeceği ve acil durumlarda ne yapılması gerektiği açıklanmıştır.\n*   **Hizmet Bedellerinin Karşılanması:** Yeşil kart kapsamında verilen sağlık hizmetlerinin, ilaçların, tetkiklerin ve diğer tıbbi giderlerin nasıl faturalandırılacağı ve ödeneceği belirtilmiştir.\n*   **Yükümlülükler ve Sorumluluklar:** Yeşil kart sahiplerinin, sağlık kuruluşlarının ve kamu görevlilerinin yetki ve sorumlulukları açıklanmıştır.\n*   **Yeşil Kartın İptali ve Geri Alınması:** Yeşil kartın hangi durumlarda geçersiz sayılacağı, geri alınabileceği ve yersiz ödemelerin nasıl geri tahsil edileceği açıklanmıştır.\n\nBu yönetmelik, maddi durumu yetersiz olan kişilerin sağlık hizmetlerinden yararlanmasını sağlamak için detaylı kurallar ve prosedürler belirlemektedir.\n",
      ],
    },
  ]
)

response = chat_session.send_message("INSERT_INPUT_HERE")

print(response.text)
