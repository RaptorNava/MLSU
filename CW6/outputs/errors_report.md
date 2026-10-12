# Анализ 5 ошибок модели

Гипотезы ниже сформированы автоматически по простым правилам. Прочитайте каждый текст и подтвердите/исправьте гипотезу вручную.

## Ошибка №1 — FP (ждали neg, предсказан pos)
- **Истина / предсказание:** negative / positive (уверенность 0.87)
- **Текст:** «Why does the poster & artwork say "Clubbed is one of the best UK indie films I have seen in a very long time. SCREEN INTERNATIONAL" when it was a quote of the French distributor REPORTED by Screen International (an influential film trade publication). See www.screendaily.com/ScreenDailyArticle.aspx?intStoryID=39811 which reads:  "Pretty Pictures has acquired all French-speaking rights to Neil Thompson's Clubbed ....James Velaise, president of Pretty Pictures, said: "Clubbed is one of the best UK...»
- **После очистки:** poster artwork say club one best uk indi film seen veri long time screen intern quot french distributor report screen intern influenti film trade public see www screendaili com screendailyarticl aspx intstoryid read pretti pictur acquir french speak right neil thompson club jame velais presid pretti...
- **Токенов после очистки:** 67, доля слов вне словаря: 0.16
- **Топ вкладов в 'positive':** [('one best', 0.386), ('best', 0.362), ('veri', 0.202), ('long time', 0.147), ('film seen', 0.135)]
- **Топ вкладов в 'negative':** [('pretti', -0.123), ('say', -0.076), ('said', -0.061), ('veri long', -0.054), ('public', -0.044)]
- **Гипотеза(ы):** отрицание/сложная конструкция: проверьте, как ведут себя биграммы с 'not'
- **Ваш вывод после прочтения:** _допишите (ирония / отрицание / шум разметки / ...)_

## Ошибка №2 — FP (ждали neg, предсказан pos)
- **Истина / предсказание:** negative / positive (уверенность 0.85)
- **Текст:** «Victor Jory never became a major star. He is better known for later character roles than for his early leads. But he was very handsome and an excellent actor.  His love scenes with Loretta Young in this romantic adventure thriller are passionate. Their kisses look very real. And very modern.  Vivienne Osborne is a standout here also. She plays a woman with a reputation. But the character has a good heart.  The print I saw was not clear. But what a joy that rare movies like this are turning up! A...»
- **После очистки:** victor jori never becam major star better known later charact role earli lead but veri handsom excel actor love scene loretta young romant adventur thriller passion kiss look veri real veri modern vivienn osborn standout also play woman reput but charact good heart print saw not clear but joy rare m...
- **Токенов после очистки:** 75, доля слов вне словаря: 0.21
- **Топ вкладов в 'positive':** [('veri', 0.318), ('excel', 0.284), ('love', 0.208), ('still', 0.13), ('heart', 0.11)]
- **Топ вкладов в 'negative':** [('noth', -0.235), ('look', -0.093), ('better', -0.09), ('watchabl', -0.088), ('sort', -0.075)]
- **Гипотеза(ы):** отрицание/сложная конструкция: проверьте, как ведут себя биграммы с 'not'
- **Ваш вывод после прочтения:** _допишите (ирония / отрицание / шум разметки / ...)_

## Ошибка №3 — FN (ждали pos, предсказан neg)
- **Истина / предсказание:** positive / negative (уверенность 0.91)
- **Текст:** «I must say, I thought I had seen it all. I am an extremely jaded movie buff. This movie didn't shock me, by any means. I'm way past that point. But it did take me to certain emotional places I didn't know I could go to. I had no idea I could ever find (ick) the idea of beastiality erotic. Never never never. Ever. Ever. But there you go. He did it. I have to give the director credit. He pulled it off.  For the first 40 minutes this movie is a TOTAL bore. We start off with some very explicit foota...»
- **После очистки:** must say thought seen extrem jade movi buff movi not shock mean way past point but take certain emot place not know could go no idea could ever find ick idea beastial erot never never never ever ever but go give director credit pull first minut movi total bore start veri explicit footag two hors sex...
- **Токенов после очистки:** 162, доля слов вне словаря: 0.12
- **Топ вкладов в 'positive':** [('beauti', 0.137), ('veri', 0.132), ('recommend', 0.107), ('definit', 0.104), ('good', 0.097)]
- **Топ вкладов в 'negative':** [('minut', -0.593), ('bore', -0.185), ('idea', -0.176), ('tri', -0.145), ('could', -0.136)]
- **Гипотеза(ы):** отрицание/сложная конструкция: проверьте, как ведут себя биграммы с 'not'
- **Ваш вывод после прочтения:** _допишите (ирония / отрицание / шум разметки / ...)_

## Ошибка №4 — FN (ждали pos, предсказан neg)
- **Истина / предсказание:** positive / negative (уверенность 0.89)
- **Текст:** «I enjoyed this series, but felt that the whole thing was let down by the sound recording/mixing.  For whatever reason, they've had to employ an awful lot of what's called ADR, where the actors replace the original location sound with a re-recording of it in a sound booth.  The reasons for doing this are usually due to problems at the location, or because somebody screwed up somewhere in the sound production chain.  It wouldn't be so bad if the ADR was done well, but at times it's just plain dist...»
- **После очистки:** enjoy seri but felt whole thing let sound record mix whatev reason employ aw lot call adr actor replac origin locat sound record sound booth reason usual due problem locat somebodi screw somewher sound product chain would not bad adr done well but time plain distract not adr problem sound mix not ma...
- **Токенов после очистки:** 60, доля слов вне словаря: 0.13
- **Топ вкладов в 'positive':** [('enjoy', 0.202), ('well', 0.14), ('perform', 0.137), ('record', 0.095), ('seri', 0.094)]
- **Топ вкладов в 'negative':** [('aw', -0.404), ('bad', -0.402), ('reason', -0.314), ('problem', -0.291), ('wrong', -0.137)]
- **Гипотеза(ы):** отрицание/сложная конструкция: проверьте, как ведут себя биграммы с 'not'
- **Ваш вывод после прочтения:** _допишите (ирония / отрицание / шум разметки / ...)_

## Ошибка №5 — FP (ждали neg, предсказан pos)
- **Истина / предсказание:** negative / positive (уверенность 0.50)
- **Текст:** «A klutzy young man returns West after being schooled in the hotel business via Boston; he quickly learns his friends in Spanish-colonized Old California expect him to fill his deceased father's shoes instead--that of a romantic thief known for kissing his female victims after robbing them. Colorful but silly M-G-M production has a great deal of talent before and behind the camera, but it never takes off. This might have been fun, second-string material for Abbott & Costello, but Frank Sinatra lo...»
- **После очистки:** klutzi young man return west school hotel busi via boston quick learn friend spanish colon old california expect fill deceas father shoe instead romant thief known kiss femal victim rob color but silli g product great deal talent behind camera but never take might fun second string materi abbott cos...
- **Токенов после очистки:** 107, доля слов вне словаря: 0.12
- **Топ вкладов в 'positive':** [('great', 0.245), ('sinatra', 0.207), ('fun', 0.166), ('perform', 0.102), ('west', 0.079)]
- **Топ вкладов в 'negative':** [('poor', -0.294), ('instead', -0.171), ('embarrass', -0.161), ('decent', -0.09), ('might', -0.086)]
- **Гипотеза(ы):** смешанная тональность: сильные позитивные и негативные признаки уравновешивают друг друга; отрицание/сложная конструкция: проверьте, как ведут себя биграммы с 'not'
- **Ваш вывод после прочтения:** _допишите (ирония / отрицание / шум разметки / ...)_
