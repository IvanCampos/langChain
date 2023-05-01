const { OpenAI } = require("langchain/llms/openai");
const { RetrievalQAChain } = require("langchain/chains");
const { HNSWLib } = require("langchain/vectorstores/hnswlib");
const { OpenAIEmbeddings } = require("langchain/embeddings/openai");
const { RecursiveCharacterTextSplitter } = require ("langchain/text_splitter");
const fs = require("fs");

const dotenv = require("dotenv");
dotenv.config();

async function chatWithTxt() {
    // Initialize the LLM to use to answer the question.
    const model = new OpenAI({});
    const text = fs.readFileSync("../data/us-constitution.txt", "utf8");
    const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
    const docs = await textSplitter.createDocuments([text]);
  
    // Create a vector store from the documents.
    const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
  
    // Create a chain that uses the OpenAI LLM and HNSWLib vector store.
    const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());
    const res = await chain.call({
      query: "What is the 2nd ammendment?",
    });
    console.log("RESPONSE: " + res.text);
  };

chatWithTxt();
