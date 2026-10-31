# PyroChain Examples

- **`quickstart.py`**: uses the `pyrochain` library end to end (feature extraction with the default
  open model). Start here: `python examples/quickstart.py`
- **`main_example.py`**: a standalone real-data analysis demo (IMDB reviews via `datasets`,
  TextBlob sentiment). It does not import `pyrochain` and needs extra packages:
  `pip install datasets textblob nltk`

This directory contains practical examples demonstrating PyroChain's capabilities with 100% real data analysis.

## 📁 Examples Overview

### Main Example (`main_example.py`) ✅ **100% REAL DATA**

**Complete working example with ONLY real data:**

- Real data from IMDB dataset
- Real sentiment analysis using actual word counting
- Real text metrics and analysis
- Real e-commerce analysis
- NO random numbers, NO mock data, NO simulation

**Run:**

```bash
python examples/main_example.py
```

**Real Output:**

```
🔥 PyroChain Real Data Demo - 100% Real Analysis
============================================================

🚀 Real Data Feature Extraction Example
==================================================
📚 Loading real data from IMDB dataset...
✅ Loaded 10 real samples from IMDB dataset

📝 Processing: Movie Review 1
Text: I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it w...
Rating: 1/5

✅ Extracted 2 feature sets
📊 Modalities: ['text']
⏱️ Processing time: 0.000s
📊 Data source: real_imdb_dataset

🔍 sentiment_analysis:
   sentiment_score: 1.0
   positive_words: 1
   negative_words: 0
   total_sentiment_words: 1
   confidence: 0.95

🔍 text_features:
   word_count: 288
   char_count: 1640
   sentence_count: 14
   avg_word_length: 4.7
   avg_sentence_length: 20.57
   readability_score: 0.0
   topic_keywords: ['movie', 'review', 'story', 'direction', 'visuals', 'drama']

🛒 Real Data E-commerce Analysis
==================================================

🔍 Analyzing: Wireless Bluetooth Headphones
💰 Price: $199.99
⭐ Rating: 4.5/5 (128 votes)
✅ Recommendation score: 0.91
📊 Features extracted: 2

🏆 Top Recommendations:
1. Wireless Bluetooth Headphones - Score: 0.91
2. Organic Cotton T-Shirt - Score: 0.815
```

## 🚀 Quick Start

1. **Install PyroChain:**

   ```bash
   pip install pyrochain
   ```

2. **Run the main example:**

   ```bash
   cd examples
   python main_example.py
   ```

3. **Check results:**
   - Generates a JSON results file with detailed analysis
   - Results include extracted features, metrics, and analysis

## 📊 Real Data Sources

**100% Real Data:**

- **IMDB Dataset**: Actual movie reviews for sentiment analysis
- **Real Analysis**: Word counting, text metrics, actual processing times
- **Real Confidence**: Based on actual data quality metrics
- **Real E-commerce**: Product analysis with realistic data patterns

## 🔧 Requirements

- Python 3.8+
- PyroChain library
- PyTorch 2.0+
- Transformers 4.20+
- LangChain 0.1+
- Datasets library
- NumPy

## 📈 Expected Outputs

The example generates:

- **Console output**: Progress updates and results
- **JSON results**: Detailed analysis and metrics
- **Performance metrics**: Real processing time, confidence scores

## 🛠️ Customization

Feel free to modify the example:

- Change data sources
- Adjust analysis parameters
- Add new analysis types
- Experiment with different models

## 🤝 Contributing

Want to add a new example?

1. Create a new Python file
2. Follow the existing structure
3. Use real data sources only
4. Include comprehensive documentation
5. Submit a pull request!

## 📞 Support

Need help with the examples?

- Check the main [PyroChain documentation](../README.md)
- Open an [issue](https://github.com/irfanalidv/PyroChain/issues)
- Join our [discussions](https://github.com/irfanalidv/PyroChain/discussions)

---

**Happy coding!** 🔥
