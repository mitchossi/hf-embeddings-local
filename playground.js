// a playground for other features of transformers.js 
// short overview: https://huggingface.co/docs/transformers.js/pipelines
import { pipeline } from "@xenova/transformers";
import wavefile from "wavefile";
import fs from "fs";


// Get sentiment of text
async function sentiment() {
    const classifier = await pipeline('sentiment-analysis', 'Xenova/bert-base-multilingual-uncased-sentiment');
    let result = await classifier(['I love transformers!', 'I hate transformers!', 'Transformers are ok.']);

    console.log("sentiment", result);
}

// Audio File to Text
async function speechRecognition() {
    const audioBuffer = fs.readFileSync('./audio/nonsense.wav');
    
    //"the Web Audio API is not available in Node.js, meaning we can’t use the AudioContext class to process audio."
    // => workaround with wavefile needed (https://huggingface.co/docs/transformers.js/guides/node-audio-processing)
    
    // Workaround start
    // Read audio file and convert it to required format
    const wav = new wavefile.WaveFile(audioBuffer);
    wav.toBitDepth('32f'); // Pipeline expects input as a Float32Array
    wav.toSampleRate(16000); // Whisper expects audio with a sampling rate of 16000

    const audioData = wav.getSamples();
    if (Array.isArray(audioData)) {
        if (audioData.length > 1) {
          const SCALING_FACTOR = Math.sqrt(2);
      
          // Merge channels (into first channel to save memory)
          for (let i = 0; i < audioData[0].length; ++i) {
            audioData[0][i] = SCALING_FACTOR * (audioData[0][i] + audioData[1][i]) / 2;
          }
        }
      
        // Select first channel
        audioData = audioData[0];
      }
      // Workaround end




    let transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-small.en');
    
    // Transcribe an audio file, loaded from a URL.
    let result = await transcriber(audioData);


    console.log("speechRecognition", result);

}

// Generate a text from an input text
async function text2text(inputText) {

    const textGenerator = await pipeline('text2text-generation', 'Xenova/LaMini-Flan-T5-783M');
        let result = await textGenerator(inputText, {
        max_new_tokens: 4000,
        temperature: 0.9,
        repetition_penalty: 2.0,
        no_repeat_ngram_size: 3,

        // top_k: 20,
        // do_sample: true,
    });

    console.log("text2text", result);
}


// await sentiment();
// await speechRecognition();

// * a bit more playing with text2text * 

await text2text(`Tell me a joke about fishes and cats`);
// => Tells laways the same Joke:  'Why did the fish and cat team up to make a joke? Because they had great camouflage skills!'

// await text2text(`How many planets are there in the solar system?`);
// =>  answers correctly

// quick note: I'm using false to disable calling the function  (as an alternative to comments)
const llmTxt = fs.readFileSync("./texts/llm.txt", "utf8"); // short text about large language model
false && await text2text(`
    Given the following text: ${llmTxt}}. Answer the following question:
    "What are some of the ethical implications and challenges associated with large language models?"`
);
// => doesn't work that bad

false && await text2text(`
    The following is a conversation between You and a User which you shall continue:

    User: "What is a vector database?"
    You: "A vector database is a collection of data that includes numerical values representing the positions and orientations of objects in space."
    User: "Can you give me an example?"
    You: "Sure, an example of a vector database is the GIS (Global Navigation Satellite Systems) dataset."
    USER: "Please explain your last message in more detail."
`)
// => Since this isn't a "Conversational" Model the above instruction doesn't work perfectly but not too bad