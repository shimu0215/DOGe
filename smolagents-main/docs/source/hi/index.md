# `smolagents`

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/license_to_call.png" width=100%/>
</div>

यह लाइब्रेरी पावरफुल एजेंट्स बनाने के लिए सबसे सरल फ्रेमवर्क है! वैसे, "एजेंट्स" हैं क्या? हम अपनी परिभाषा [इस पेज पर](conceptual_guides/intro_agents) प्रदान करते हैं, जहाँ आपको यह भी पता चलेगा कि इन्हें कब उपयोग करें या न करें (स्पॉइलर: आप अक्सर एजेंट्स के बिना बेहतर काम कर सकते हैं)।

यह लाइब्रेरी प्रदान करती है:

✨ **सरलता**: Agents का लॉजिक लगभग एक हजार लाइन्स ऑफ़ कोड में समाहित है। हमने रॉ कोड के ऊपर एब्स्ट्रैक्शन को न्यूनतम आकार में रखा है!

🌐 **सभी LLM के लिए सपोर्ट**: यह हब पर होस्ट किए गए मॉडल्स को उनके `transformers` वर्जन में या हमारे इन्फरेंस API के माध्यम से सपोर्ट करता है, साथ ही OpenAI, Anthropic से भी... किसी भी LLM से एजेंट को पावर करना वास्तव में आसान है।

🧑‍💻 **कोड Agents के लिए फर्स्ट-क्लास सपोर्ट**, यानी ऐसे एजेंट्स जो अपनी एक्शन्स को कोड में लिखते हैं (कोड लिखने के लिए उपयोग किए जाने वाले एजेंट्स के विपरीत), [यहाँ और पढ़ें](tutorials/secure_code_execution)।

🤗 **हब इंटीग्रेशन**: आप टूल्स को हब पर शेयर और लोड कर सकते हैं, और आगे और भी बहुत कुछ आने वाला है!
!

<div class="mt-10">
  <div class="w-full flex flex-col space-y-4 md:space-y-0 md:grid md:grid-cols-2 md:gap-y-4 md:gap-x-5">
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./guided_tour"
      ><div class="w-full text-center bg-gradient-to-br from-blue-400 to-blue-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">गाइडेड टूर</div>
      <p class="text-gray-700">बेसिक्स सीखें और एजेंट्स का उपयोग करने में परिचित हों। यदि आप पहली बार एजेंट्स का उपयोग कर रहे हैं तो यहाँ से शुरू करें!</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./examples/text_to_sql"
      ><div class="w-full text-center bg-gradient-to-br from-indigo-400 to-indigo-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">हाउ-टू गाइड्स</div>
      <p class="text-gray-700">एक विशिष्ट लक्ष्य प्राप्त करने में मदद के लिए गाइड: SQL क्वेरी जनरेट और टेस्ट करने के लिए एजेंट बनाएं!</p>
    </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./conceptual_guides/intro_agents"
      ><div class="w-full text-center bg-gradient-to-br from-pink-400 to-pink-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">कॉन्सेप्चुअल गाइड्स</div>
      <p class="text-gray-700">महत्वपूर्ण विषयों की बेहतर समझ बनाने के लिए उच्च-स्तरीय व्याख्याएं।</p>
   </a>
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./tutorials/building_good_agents"
      ><div class="w-full text-center bg-gradient-to-br from-purple-400 to-purple-500 rounded-lg py-1.5 font-semibold mb-5 text-white text-lg leading-relaxed">ट्यूटोरियल्स</div>
      <p class="text-gray-700">एजेंट्स बनाने के महत्वपूर्ण पहलुओं को कवर करने वाले क्ट्यूटोरियल्स।</p>
    </a>
  </div>
</div>
