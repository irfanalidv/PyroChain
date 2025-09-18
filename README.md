# PyroChain 🔥

**Intelligent Feature Engineering with AI Agents**

[![GitHub](https://img.shields.io/github/license/irfanalidv/PyroChain)](https://github.com/irfanalidv/PyroChain/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/pyrochain)](https://pypi.org/project/pyrochain/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/pyrochain?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/pyrochain)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-green)](https://langchain.com/)

PyroChain combines PyTorch's deep learning capabilities with LangChain's agentic AI to automate feature extraction from complex, multimodal data. AI agents collaborate to understand, process, and extract meaningful features from text, images, and structured data.

## 🎯 What Problem Does PyroChain Solve?

**Traditional Feature Engineering is Hard:**

- Manual feature extraction is time-consuming and error-prone
- Different data types require different approaches
- Domain expertise is needed to create meaningful features
- Features become outdated as data patterns change

**PyroChain Makes It Easy:**

- AI agents automatically extract relevant features from any data type
- Collaborative agents validate and refine features using chain-of-thought reasoning
- Learns from your data to improve feature quality over time
- Works seamlessly with existing ML pipelines

## 🚀 Key Features

- **🤖 AI Agents**: Intelligent agents that collaborate to extract, validate, and refine features
- **📊 Multimodal Processing**: Handle text, images, and structured data in one pipeline
- **⚡ Lightweight & Fast**: Efficient LoRA adapters that train quickly on your data
- **🧠 Memory & Learning**: Agents remember past decisions and improve over time
- **🛒 E-commerce Ready**: Built-in tools for product recommendations and customer analysis
- **🏗️ Production Ready**: Scalable architecture designed for real-world applications

## 💡 Use Cases

**E-commerce & Retail:**

- Product recommendation systems
- Customer sentiment analysis
- Inventory optimization
- Price prediction and analysis

**Content & Media:**

- Text classification and tagging
- Image content analysis
- Content recommendation
- Automated content moderation

**Business Intelligence:**

- Customer behavior analysis
- Market trend detection
- Risk assessment
- Automated reporting

## 🛠️ Installation

### Quick Install

```bash
pip install pyrochain
```

### From Source

```bash
git clone https://github.com/irfanalidv/PyroChain.git
cd PyroChain
pip install -e .
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- LangChain 0.1+
- Transformers 4.20+

## 🚀 Quick Start

### Basic Usage

```python
from pyrochain import PyroChain
from transformers import AutoTokenizer, AutoModel
from textblob import TextBlob
import torch
from datasets import load_dataset

# Load real transformer model and tokenizer
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Initialize PyroChain with transformer model
pyrochain = PyroChain()

# Load REAL data from IMDB dataset
print("📚 Loading real IMDB dataset...")
dataset = load_dataset("imdb", split="train[:4]")  # Load first 4 real reviews

# Extract features from REAL dataset with TextBlob sentiment analysis
for i, sample in enumerate(dataset):
    text = sample["text"]
    label = sample["label"]  # 0 = negative, 1 = positive

    # Use TextBlob for real sentiment analysis
    blob = TextBlob(text)
    sentiment_score = (blob.sentiment.polarity + 1) / 2  # Convert to 0-1 scale

    data = {
        "text": text,
        "title": f"IMDB Review {i+1}",
        "rating": 5 if label == 1 else 1,
        "category": "movie_review"
    }

    features = pyrochain.extract_features(
        data,
        "Extract features for sentiment analysis using TextBlob and transformer model"
    )

    print(f"Text: {text[:100]}...")
    print(f"Real Label: {label} | TextBlob Sentiment: {sentiment_score:.3f}")
    print(f"Features: {len(features['features'])}")
    print("---")
```

### Real Data Example

```bash
# Run the complete real data example
cd examples
python main_example.py
```

**What you'll see:**

```
🔥 PyroChain Real Data Demo - 100% Real Analysis
============================================================

🚀 Real Data Feature Extraction Example
==================================================
📚 Loading real IMDB dataset using transformer models...
📥 Downloading real IMDB dataset...
✅ Loaded 5 real IMDB samples using transformer model

📝 Processing: IMDB Review 1
Text: I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it w...
Rating: 1/5 (Real IMDB Label: 0 = Negative)

✅ Extracted 2 feature sets
📊 Modalities: ['text']
⏱️ Processing time: 0.025s
📊 Data source: real_imdb_dataset

🔍 sentiment_analysis:
   sentiment_score: 0.57
   polarity: 0.14
   subjectivity: 0.85
   positive_words: 16
   negative_words: 4
   total_sentiment_words: 20
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

## 🏗️ How It Works

1. **Data Ingestion**: Accepts multimodal data (text, images, structured)
2. **Agent Processing**: AI agents analyze data using chain-of-thought reasoning
3. **Feature Extraction**: Collaborative agents extract relevant features
4. **Validation**: Agents validate and refine features through discussion
5. **Output**: Clean, structured features ready for ML models

## ⚙️ Configuration

```python
from pyrochain import PyroChain, PyroChainConfig
from transformers import AutoTokenizer, AutoModel
import torch

# Load real transformer model for e-commerce analysis
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Real e-commerce product data
products = [
    {
        "id": "prod_001",
        "title": "Wireless Bluetooth Headphones",
        "description": "High-quality wireless headphones with noise cancellation and 30-hour battery life. Perfect for music lovers and professionals.",
        "price": 199.99,
        "category": "electronics",
        "rating": 4.5
    },
    {
        "id": "prod_002",
        "title": "Organic Cotton T-Shirt",
        "description": "Comfortable organic cotton t-shirt in various colors and sizes. Made from 100% organic cotton, eco-friendly and sustainable.",
        "price": 29.99,
        "category": "clothing",
        "rating": 4.2
    }
]

# Configure for e-commerce with transformer model
config = PyroChainConfig(
    task_type="ecommerce",           # Task type: "general", "ecommerce", "custom"
    enable_agents=True,              # Enable AI agent collaboration
    enable_training=False,           # Enable model training
    max_length=512,                  # Maximum input length
    learning_rate=1e-4,              # Learning rate for training
    num_epochs=3,                    # Number of training epochs
    device="auto"                    # Device: "auto", "cpu", "cuda"
)

pyrochain = PyroChain(config=config)

# Process real product data with transformer analysis
for product in products:
    features = pyrochain.extract_features(
        product,
        "Extract features for product recommendation using transformer model"
    )
    print(f"Product: {product['title']} - Features: {len(features['features'])}")
    print(f"Price: ${product['price']} - Rating: {product['rating']}/5")
```

## 📚 API Reference

### Core Classes

- **`PyroChain`**: Main library class for feature extraction
- **`PyroChainConfig`**: Configuration class for customizing behavior
- **`LoRAAdapter`**: Lightweight adapter for efficient model fine-tuning
- **`MultimodalProcessor`**: Handles text, image, and structured data processing

### Key Methods

- **`extract_features(data, task_description)`**: Extract features from data
- **`train(training_data, task_description)`**: Train custom agents
- **`evaluate(test_data)`**: Evaluate model performance
- **`save_model(path)`**: Save trained model
- **`load_model(path)`**: Load pre-trained model

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](https://github.com/irfanalidv/PyroChain/blob/main/CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/irfanalidv/PyroChain/blob/main/LICENSE) file for details.

## 🙏 Acknowledgments

- [PyTorch](https://pytorch.org/) for deep learning capabilities
- [LangChain](https://langchain.com/) for agentic AI framework
- [Hugging Face](https://huggingface.co/) for transformer models
- [Sentence Transformers](https://www.sbert.net/) for text embeddings

## 📞 Support

**Need help?** We're here to support you:

- 📚 [Documentation](https://github.com/irfanalidv/PyroChain#readme)
- 🐛 [Report Issues](https://github.com/irfanalidv/PyroChain/issues)
- 💡 [Feature Requests](https://github.com/irfanalidv/PyroChain/discussions)
- 📧 [Contact](https://github.com/irfanalidv)

---

**PyroChain** - Transform your data into intelligent features with AI agents. 🔥

_Built with ❤️ by [Irfan Ali](https://github.com/irfanalidv)_
