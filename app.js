const express = require("express");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const admin = require('firebase-admin');
const pdf = require('pdf-parse');
const { chunk } = require('llm-chunk');
const dotenv = require('dotenv');
const path = require('path');
const fs = require('fs').promises;
const cosineSimilarity = require('cosine-similarity');
const cors = require('cors');

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));
app.use('/views', express.static('views'));

// Inisialisasi Firebase
const serviceAccount = require('./config/your_firebase_service_account');
admin.initializeApp({
  credential: admin.credential.cert(serviceAccount)
});

const db = admin.firestore();

// Inisialisasi Google AI
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const embeddingModel = genAI.getGenerativeModel({ model: "embedding-001" });
const generativeModel = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

// Konfigurasi Chunking
const chunkingConfig = {
  minLength: 1000,
  maxLength: 2000,
  splitter: 'sentence',
  overlap: 100,
  delimiters: '',
};

// Fungsi untuk mendapatkan semua file PDF dari direktori
async function getPdfFiles(directoryPath) {
  try {
    const files = await fs.readdir(directoryPath);
    return files
      .filter(file => path.extname(file).toLowerCase() === '.pdf')
      .map(file => path.join(directoryPath, file));
  } catch (error) {
    console.error('Gagal membaca direktori PDF:', error);
    return [];
  }
}

// Fungsi Ekstraksi Teks dari PDF
async function extractTextFromPdf(filePath) {
  try {
    const dataBuffer = await fs.readFile(filePath);
    const data = await pdf(dataBuffer);
    return data.text;
  } catch (error) {
    console.error('Gagal mengekstrak teks dari PDF:', error);
    throw error;
  }
}

// Fungsi Indexing PDF ke Firestore
async function indexPdfDocuments(directoryPath, collectionName = 'document_chunks') {
  try {
    // Hapus dokumen lama
    const oldDocsSnapshot = await db.collection(collectionName).get();
    const batch = db.batch();
    oldDocsSnapshot.docs.forEach(doc => batch.delete(doc.ref));
    await batch.commit();

    // Ambil semua file PDF
    const pdfFiles = await getPdfFiles(directoryPath);
    console.log(`Menemukan ${pdfFiles.length} file PDF untuk diindeks`);

    // Proses setiap PDF
    for (const filePath of pdfFiles) {
      // Ekstrak teks dari PDF
      const pdfText = await extractTextFromPdf(filePath);

      // Chunk teks
      const chunks = await chunk(pdfText, chunkingConfig);

      // Buat embedding untuk setiap chunk
      const batchOperations = chunks.map(async (chunkText) => {
        const embeddingResponse = await embeddingModel.embedContent({
          content: { parts: [{ text: chunkText }] }
        });

        return {
          text: chunkText,
          embedding: embeddingResponse.embedding.values,
          source: path.basename(filePath)
        };
      });

      // Tunggu semua embedding selesai
      const documentChunks = await Promise.all(batchOperations);

      // Simpan ke Firestore
      const batchWrite = db.batch();
      documentChunks.forEach((doc) => {
        const docRef = db.collection(collectionName).doc();
        batchWrite.set(docRef, doc);
      });

      await batchWrite.commit();
      console.log(`Berhasil mengindeks ${documentChunks.length} chunk dari ${path.basename(filePath)}`);
    }
  } catch (error) {
    console.error('Gagal mengindeks dokumen:', error);
  }
}

// Fungsi Retrieval dengan Cosine Similarity
async function retrieveRelevantDocuments(query, k = 5, collectionName = 'document_chunks') {
  try {
    // Buat embedding untuk query
    const queryEmbedding = await embeddingModel.embedContent({
      content: { parts: [{ text: query }] }
    });

    // Ambil semua dokumen dari Firestore
    const snapshot = await db.collection(collectionName).get();
    const documents = snapshot.docs.map(doc => ({
      id: doc.id,
      ...doc.data()
    }));

    // Hitung similaritas
    const similarities = documents.map(doc => ({
      ...doc,
      similarity: cosineSimilarity(queryEmbedding.embedding.values, doc.embedding)
    }));

    // Urutkan berdasarkan similaritas
    similarities.sort((a, b) => b.similarity - a.similarity);

    // Kembalikan top-k dokumen
    return similarities.slice(0, k);
  } catch (error) {
    console.error('Gagal mengambil dokumen relevan:', error);
    return [];
  }
}

// Rute Utama
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'views', 'index.html'));
});

// Rute untuk Indexing PDF
app.get('/index-pdfs', async (req, res) => {
  const pdfDirectory = 'D:\\UPI\\Sem 5\\Inkubator\\Chatbot\\Chatbot_Umum\\knowledge';
  
  try {
    await indexPdfDocuments(pdfDirectory);
    res.json({ message: 'PDF berhasil diindeks' });
  } catch (error) {
    res.status(500).json({ error: 'Gagal mengindeks PDF' });
  }
});

// Rute untuk RAG Query
// Tambahkan atau modifikasi route di server.js  
app.post("/chat", async (req, res) => {  
  const userInput = req.body.userInput;  
  const startTime = Date.now();  

  try {  
    // Ambil dokumen relevan  
    const relevantDocs = await retrieveRelevantDocuments(userInput, 5);  

    // Gabungkan dokumen untuk konteks  
    const context = relevantDocs.map(doc => doc.text).join('\n\n');  

    // Buat prompt augmented  
    const augmentedPrompt = `  
Konteks: ${context}  

Pertanyaan: ${userInput}  

Berdasarkan konteks yang diberikan, jawab pertanyaan dengan detail dan akurat.   
Jika informasi tidak cukup atau tidak relevan, katakan "Maaf, saya tidak menemukan informasi yang spesifik untuk menjawab pertanyaan Anda."  
    `;  

    // Generate jawaban  
    const chatSession = generativeModel.startChat({  
      generationConfig: {  
        temperature: 0.7,  
        maxOutputTokens: 1024  
      }  
    });  

    const result = await chatSession.sendMessage(augmentedPrompt);  
    const responseText = result.response.text();  

    const processingTime = Date.now() - startTime;  

    res.json({   
      response: responseText,  
      metadata: {  
        processingTime: processingTime,  
        retrievedDocs: relevantDocs.length  
      }  
    });  

  } catch (error) {  
    console.error('Gagal memproses chat', {   
      error: error.message,  
      userInput,  
      stack: error.stack   
    });  
    res.status(500).json({   
      error: "Terjadi kesalahan dalam memproses percakapan",  
      details: error.message   
    });  
  }  
});

app.listen(port, async () => {
  console.log(`Server berjalan di http://localhost:${port}`);
  
  // Otomatis index PDF saat server start
  const pdfDirectory = 'D:\\UPI\\Sem 5\\Inkubator\\Chatbot\\Chatbot_Umum\\knowledge';
  await indexPdfDocuments(pdfDirectory);
});