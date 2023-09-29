import fs from "fs";
import { dotProduct } from "./helper/dotProduct.js";
export async function moreEmbeddings(extractor, extractorOptions) {
    console.log("\n\n### moreEmbeddings called ###");

    // question which will be checked for similarity to the other texts
    let question = "What is the capital of France?";
    question = "What are some of the ethical implications and challenges associated with large language models?";
    question = "In what fields are vector databases particularly useful and why?";
    question = "Where to store data which is supposed to be used for a similarity search?";


    // the texts to which the question will be compared
    const vectorDbTxt = fs.readFileSync("./texts/vector-db.txt", "utf8"); // short text about vector database
    const llmTxt = fs.readFileSync("./texts/llm.txt", "utf8"); // short text about large language model
    // console.log("vectorDbTxt", vectorDbTxt);


    // create embeddings for the texts and the question
    const vectorDbEmbedding = await extractor(vectorDbTxt, extractorOptions);
    const llmEmbedding = await extractor(llmTxt, extractorOptions);
    const questionEmbedding = await extractor(question, extractorOptions);

    // check similiraty to vectorDbTxt
    const similarityToVectorDb = dotProduct(questionEmbedding.data,vectorDbEmbedding.data);
    // check similiraty to vectorDbTxt
    const similarityLlm = dotProduct(questionEmbedding.data,llmEmbedding.data);

    console.log(`
    The question 
    "${question}" has 
    a similarity of ${similarityToVectorDb} to the text "Vector Database Text"
    and a similarity of ${similarityLlm} to the text "Large Language Model Text"`);
    

}
