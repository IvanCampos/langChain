import * as fs from "fs";
import { OpenAI } from "langchain/llms/openai";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { RetrievalQAChain, LLMChain } from "langchain/chains";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import {
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
  ChatPromptTemplate,
} from "langchain/prompts";

import dotenv from "dotenv";
dotenv.config();

const green = "\x1b[32m";
const reset = "\x1b[0m";

async function translateToBostonAccent(): Promise<void> {
  const translationPrompt = ChatPromptTemplate.fromPromptMessages([
    SystemMessagePromptTemplate.fromTemplate(
      "You are a helpful assistant that translates {input_language} to English Boston Accent."
    ),
    HumanMessagePromptTemplate.fromTemplate("{text}"),
  ]);

  const chat = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0,
  });

  const chain = new LLMChain({
    prompt: translationPrompt,
    llm: chat,
  });

  const response = await chain.call({
    input_language: "English",
    text: "Do you want to visit the Harbor or Harvard Yard",
  });

  let answer = JSON.stringify(response.text);
  console.log(`${green}${answer}${reset}`);
  console.log("\n");
}

async function chatWithTxt(): Promise<void> {
  const model = new OpenAI({modelName: "gpt-3.5-turbo"});
  const text = fs.readFileSync("../data/us-constitution.txt", "utf8");
  const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
  const docs = await textSplitter.createDocuments([text]);

  const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());

  const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());
  const res = await chain.call({
    query: "What is the 2nd ammendment?",
  });

  let answer = res.text;
  console.log(`${green}${answer}${reset}`);
  console.log("\n");
}

async function chatWithPDF(): Promise<void> {
  const model = new OpenAI({modelName:"gpt-3.5-turbo"});
  const loader = new PDFLoader("../data/pkd-metz.pdf", { splitPages: true });
  const docs = await loader.load();

  const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());

  const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());
  const res = await chain.call({
    query: "are we living in a computer generated simulation",
  });

  let answer = res.text;
  console.log(`${green}${answer}${reset}`);
}

translateToBostonAccent();
chatWithTxt();
chatWithPDF();