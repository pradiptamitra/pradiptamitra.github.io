---
title: "AI v. Tagore"
author: "Pradipta Mitra"
---

Sarvam AI has garnered well-deserved attention for their recent sequence of "[drops](https://x.com/pratykumar/status/2018027623973278107?s=46)" of ML models for Indic languages (Bangla included). I've used and recommended their amazing [dubbing model](https://www.sarvam.ai/blogs/sarvam-dub) myself.

The growth of sovereign models is to be lauded — but as the wise say, there is no growth without pain (Aristotle? Andrew Huberman? I forget). So I decided to dole out some, and confront their OCR model with the 800 pound gorilla: the handwritten manuscripts of Rabindranath Tagore, the great poet of Bengal. These are filled with corrections and overwriting, the crossed-out words linked and transformed into drawings.

In their [blog](https://www.sarvam.ai/blogs/Sarvam-vision) introducing the model, Sarvam reports it beating Gemini 2.0 Pro and others — 92.61 vs 90.79 character accuracy on Bengali. I took Gemini as the natural second model to run these through.

![Tagore manuscript](/assets/images/Tagore_manuscript6_c.jpg)

---

## **Round 1: A Canonical Song**

I began with a well-known Tagore song (the manuscript you see above) –

**বিধির বাঁধন কাটবে তুমি এমন শক্তিমান (You think you will break the bonds of Fate?)**

Here's the full song:

<pre>
বিধির বাঁধন কাটবে তুমি এমন শক্তিমান--
              তুমি কি   এমন শক্তিমান!
আমাদের   ভাঙাগড়া তোমার হাতে এমন অভিমান--
              তোমাদের   এমনি অভিমান ॥
     চিরদিন টানবে পিছে,   চিরদিন   রাখবে নীচে--
     এত বল   নাই রে তোমার,   সবে না সেই টান ॥
শাসনে   যতই ঘেরো   আছে বল   দুর্বলেরও,
     হও-না   যতই বড়ো   আছেন ভগবান।
          আমাদের   শক্তি মেরে   তোরাও   বাঁচবি নে রে,
              বোঝা তোর   ভারী হলেই ডুববে তরীখান।
</pre>

### **Sarvam Vision OCR Output**

The Sarvam model deviated immediately. The first lines were transcribed as:

<pre>
বিবিধ ধন্য কলার ফুল
এবং পার্সার!
তুমি কি এমন পার্সার।
</pre>

Too garbled to translate, but I cannot fail to note that it hilariously includes the transliteration of the word "parser" (পার্সার). The full output is [below](#sarvam-full-output).

Classic autoregressive drift: once a few early tokens are wrong, subsequent predictions are conditioned on the corrupted prefix, and error compounds from there. But where does পার্সার come from — is it a chimera cooked up by tokenization?

---

### **Gemini Output**

Gemini doesn't appear to have a separate OCR model. You simply provide the image to the Gemini chat interface and ask it to read it.

Gemini started off with: "This image is a famous example of **Rabindranath Tagore's** manuscript art. He famously turned his crossed-out words and corrections into flowing, organic doodles…" — before providing the transcription and then a historical exposition about the song. This is concerning: one wonders if it simply has the poem in its parametric memory and is reciting rather than reading.

Gemini's transcription starts out well:

<pre>
বিধির বাঁধন কাটবে তুমি

এমন শক্তিমান!

তুমি কি এমন শক্তিমান।

আমাদের ভাঙাগড়া তোমার হাতে

এমন অভিমান—

তোমাদের এমন অভিমান!
</pre>

But it misses a crucial poetic gesture. After using এমন (meaning "such" – e.g. "such strength") thrice, the Bard uses the variation এমনি in the last line. Gemini keeps using এমন.

Is this simply Gemini reciting a poem it knows? The transcription errors suggest otherwise:

<pre>
আমাদের পড়ি মার
তোরা বাঁধবি নে রে…
</pre>

This is incorrect — it should be:

<pre>
আমাদের   শক্তি মেরে
তোরাও   বাঁচবি নে রে,
</pre>

These are not the mistakes of a model retrieving from memory. They look like visual misreads. I strongly suspect, however, that Gemini's upfront identification of the poem's context and author is helping it tap into the right language model prior — giving it a Tagore-shaped scaffold to decode against.

## **Round 2: A Less Common Poem**

Next up: a more obscure poem — মন যে বলে, চিনি চিনি (My heart believes: I know you — I know you) — one I had definitely never read before, with a manuscript even harder to decipher than the first.

![Tagore manuscript - Mon je bole](/assets/images/mon-je-bole.jpeg)

The manuscript contains wording differences relative to the standard published version. For example, the standard published text has the more specific "চৈত্ররাতের" (april-night), where the manuscript gives the more generic "বসন্তের" (spring). The standard text is as follows:

<pre>
মন যে বলে, চিনি চিনি যে-গন্ধ বয় এই সমীরে।
কে ওরে কয় বিদেশিনী চৈত্ররাতের চামেলিরে॥
রক্তে রেখে গেছে ভাষা,
স্বপ্নে ছিল যাওয়া-আসা--
কোন্‌ যুগে কোন্‌ হাওয়ার পথে, কোন্‌ বনে কোন্‌ সিন্ধুতীরে।
এই সুদূরে পরবাসে
ওর বাঁশি আজ প্রাণে আসে।
মোর পুরাতন দিনের পাখি
ডাক শুনে তার উঠল ডাকি,
চিত্ততলে জাগিয়ে তোলে অশ্রুজলের ভৈরবীরে॥
</pre>

This allows a clearer test of whether a model reads the image or defaults to a canonical variant.

### **Sarvam Output**

First up, Sarvam. Unfortunately equally poor — mistakes start on the 4th word and it's all downhill from there:

<pre>
মন যে বলে চিন চিনি
যে সঙ্গী কর এই সমীক্ষা।
কে ওকে কম বিভাশনী
সময়ের এ সময় চেলির।
</pre>

Same pattern as before: early errors cascade. The model never recovers a foothold.

### **Gemini Output**

The first line was accurate:

মন যে বলে চিনি চিনি যে গন্ধ বয় এই সমীরে।

The output is worse after that: 

<pre>
কে ওগো রূপ-বিনোদিনী বকুলের এই চামেলিরে।
অজানা নয় আমার কাছে ভাবনাতে তার পিছু আছে, 
কোথায় ছিল এই পাখি কি জানি কার মিষ্টি হাসি। 
কোন বসন্তে সেই আঁখি দেখা কি বা কার পাশে। 
হেথায় যখন একা বসি মন প্রবাসের বাঁশি বাজে। 
শ্রাবণ দিনের ঘন মেঘ শুনে তার ধ্বনি একি আজিকে হলে বিস্মৃতি তন্দ্রাতলের বৈরাগিনী।
</pre>

This is not correct, but it's more coherent. And interestingly, it can be described stylistically as **Pseudo-Tagore** — perhaps satisfactory to a mere AGI, but not to I (you see what I did there?). If you will allow me...

মোর পুরাতন দিনের পাখি ডাক শুনে তার উঠল ডাকি — becomes — শ্রাবণ দিনের ঘন মেঘ শুনে তার ধ্বনি একি

The funny part is শ্রাবণ — the second month of the rainy season, a word Tagore was extremely fond of. Good try, Gemini. But if you read what you already wrote, there is বসন্ত (spring) sitting right in the middle. He's not going to mix up seasons like that.

Also কে ওগো রূপ-বিনোদিনী বকুলের এই চামেলিরে – it's not বিনোদিনী (She who delights), rather বিদেশিনী (She, the foreigner). If you were paying attention, you would realize that the opening verse — "My heart believes: I know you" — obviously means the poet is not so sure he knows her, and thus it *has* to be বিদেশিনী. Add to this the botanical fact that the addressee is the চামেলি flower, a Himalayan Jasmine not native to Bengal, and you have what lawyers call dispositive evidence. Mic drop.

---

## **Language Model**

Stepping down from the literary ramparts, I felt I'd do a small experiment on this language model prior point. I wrote, in my own handwriting, a ridiculous story about a talking monkey — but in two variants: standard Bengali, and a hotchpotch of highly Sanskritized words and idiomatic expressions from at least two dialects (let's call it "mixed mode").

![Standard Bengali handwriting](/assets/images/IMG_5318.jpeg)

<p style="text-align: center"><em>Standard Bengali image</em></p>

![Mixed mode Bengali handwriting](/assets/images/IMG_5319.jpeg)

<p style="text-align: center"><em>Mixed mode Bengali image</em></p>

On the standard Bengali, Sarvam is rather good (just one word wrong — অধিবেশক — which means nothing):

<pre>
গাছ থেকে নেমে এসে বাদরটি বলল,
"তুমি কে ? তুই বললেই আমাকে
নামতে হবে কেন?"
অধিবেশক বাঁদরকে আমি চড়ে মারতে
বাধ্য হলাম ।
</pre>

Gemini is essentially correct (althoug it fixes a minor typo I made):

<pre>
গাছ থেকে নেমে এসে বাঁদরটি বলল, "তুমি কে? বললেই আমাকে নামতে হবে কেন?"
অবিবেচক বাঁদরকে আমি চড় মারতে বাধ্য হলাম।
</pre>

On the mixed mode image, Sarvam is all over the place:

<pre>
গাছ থেকে এসে এসে শাখাযুগ কহিলেন,
"তুই কিজ ? ভুললে পরে আমাকে
গামতি হবি কেন?"
অবিমৃক্যনারী কপিবরকে মুখের ওপর-
চাটকানা মারতে বাধ্য হলেন।
</pre>

Some good parts though — it nailed two landmines: কপিবর (The august Monkey) and চাটকানা (slap).

Gemini, on the other hand, is surprisingly good:

<pre>
গাছ থেকে নেমে এসে শাখামৃগ কহিলেন, "তুই কিডা? বুরলে পরে আমারে প্রনতি হবি কেন?" অবিমৃষ্যকারী কপিকে মুখের ওপর চাটকানা মারতে বাধ্য হলেম।
</pre>

Pretty good — but there are three mistakes. The most interesting one is the replacement of dialect নামলি with Sanskrit প্রনতি (perhaps nudged to the top of the prior by similar-looking words in the surrounding text).

The experiment is a small one, but the pattern is clear: the language prior is not just a background condition — it actively shapes what a model sees. OCR is not purely vision. It never was.

Finally, congratulations to the Sarvam team for a great launch. As the Upanishads say, idaṁ sarvam (Sarvam is Here). Onward!

---

## Sarvam Full Output {#sarvam-full-output}
<pre>
বিবিধ ধন্য কলার ফুল
এবং পার্সার!
তুমি কি এমন পার্সার।
আমাদের ভালবাসা তোমার হাতে
এমন আতিমার
তোমাদের একটি সন্মান!
চিরদিন ও টারবে পিছে
পিছদিনা রাখার
এত বল পারে তোমার
সরল নয় দান!
আঁধরে যতই ঘোরে
আসিও বল মুধুরো,
অনায়াসায়
আমাদের পাড়ি মার
তোমার কাঁধিনেড়ে
কোথা তোর ভারি গলই
ডুবকে তরীখান!
চিরদিন
খুশি
সার্কাস
১১২২
</pre>


