# Locally Created Embeddings

This little script creates some **embeddings locally**  using models from **hugging face**. 

**Just run**
- `npm install`
- `npm start`

**Note**: First start can take some time since the used model(s) have to be downloaded and cached.

**Play around** with the sentences in `index.js` and the textes in the `texts` folder. The latter are used in `more-embeddings.js`

## Other stuff
- The **`playground.js`** is just some playground for other features of transformers.js
    - It's not used in `index.js` so you have to start it separately with `node playground.js`
    - I find the `text2text-generation` especially interesting ;) 
- For some reason the transformers don't work when the scripts are executed with **`bun`**
- When you try all models used in the scripts, the **node_modules** folder will **become bigger than 1.5 GB** since all the models are downloaded. The biggest model is the one for text2text generation