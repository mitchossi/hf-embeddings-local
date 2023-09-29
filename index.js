// - Docs/Tutorial: 
// -- https://www.youtube.com/watch?v=QdDoFfkVkcw
// -- https://huggingface.co/docs/transformers.js/index
// - more transformers: see playground.js

import { pipeline } from "@xenova/transformers";
import { dotProduct } from "./helper/dotProduct.js";
import { moreEmbeddings } from "./more-embeddings.js";

// * Sentences to embed and compare *
let sentence1 = "That is a very happy person which is walking in the rain while the sun is shining";
let sentence2 = "That is a very content woman who is walking in the rain while the sun is shining";

// some more sentences to test
sentence2 = "That is an unhappy dog which is running in the dirt while the sun is burning"; // similarity of 0.500
// sentence2 = "Maxim rocks at programming!"; // very low similarity of 0.0817
// sentence2 = "Das ist eine sehr glückliche Person, die im Regen spazieren geht, während die Sonne scheint"; // low similarity of 0.1472415...


// * create embeddings * 
const extractor = await pipeline(
    'feature-extraction', // name of feature (see: https://huggingface.co/docs/transformers.js/pipelines#available-tasks)
    'Xenova/all-MiniLM-L6-v2' // name of model (see: https://huggingface.co/models?pipeline_tag=feature-extraction&library=transformers.js&sort=downloads)
);
// - the model 'Xenova/all-MiniLM-L6-v2' gets downloaded automatically if it is not cached yet
//  -- see: ./node_modules/@xenova/transformers/.cache
// - prefetching or downloading in advance is also possible: https://huggingface.co/docs/transformers.js/custom_usage & https://youtu.be/QdDoFfkVkcw?si=ksHTJZuja5CxcKxp&t=3801 



const extractorOptions = {
    pooling: 'mean', //aggregate the embeddings of each token of the sentence to get a single embedding
    normalize: true  // normalize the embeddings to get unit vectors (makes dot product possible)
}

const embedding1 = await extractor(sentence1, extractorOptions);
const embedding2 = await extractor(sentence2, extractorOptions);
// console.log("embedding2.data", embedding2.data);

// check similiraty of two embeddings with dot product
const similarity = dotProduct(embedding1.data,embedding2.data);

console.log(`
    Similarity of 
    "${sentence1}" and 
    "${sentence2}"
    is ${similarity}`);



// *** more testing/playing aroung ***
moreEmbeddings(extractor, extractorOptions); // see "more-testing.js" for code


/* *** Further Information ***

MODELS
- the models like Xenova/all-MiniLM-L6-v2 have to exist in a special format. Thats why we have to use the Xenova/ prefix
- not all models are available in this format, but it's possible to convert other models:
 --  https://huggingface.co/docs/transformers.js/custom_usage#convert-your-models-to-onnx & https://youtu.be/QdDoFfkVkcw?si=7dsIQJufPjtFrr-q&t=4019 

*/