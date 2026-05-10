# LLMs for Time Series Forecasting Time series analysis is used for finance, healthcare, industrial IoT,
and many other fields. Statistical methods like ARIMA and Exponential...

### LLMs for Time Series Forecasting 

Time series analysis is used for finance, healthcare, industrial IoT, and many other fields. Statistical methods like ARIMA and Exponential Smoothing have been the primary tools for forecasting for decades. More recently, deep learning models such as Long Short-Term Memory networks (LSTMs) and Transformers have redefined the space --- now large language models (LLMs) and changing how we do time series analysis.


<figcaption>Photo by <a class="markup--anchor markup--figure-anchor" rel="photo-creator noopener" target="_blank">Modestas Urbonas</a> on <a class="markup--anchor markup--figure-anchor"


### Classical and Deep Learning Approaches to Time Series
Before the rise of machine learning, time series analysis was largely a domain of statistical models. ARIMA (AutoRegressive Integrated Moving Average) provided a way to model linear dependencies, while seasonal decomposition techniques helped uncover underlying trends. Economists and engineers relied on these methods to forecast stock prices, optimize supply chains, and monitor equipment failures.

Deep learning changed the game. LSTMs and GRUs (Gated Recurrent Units) introduced the ability to capture long-range dependencies in sequential data, making them useful for time series forecasting. CNNs (Convolutional Neural Networks) found applications in irregularly sampled data, while Transformers like Informer and TimeNet extended attention mechanisms to improve efficiency in long time series forecasting. These approaches significantly outperformed classical methods, but they required large labeled datasets, careful feature engineering, and domain-specific model tuning.

### Large Language Models (LLMs) in Data Analysis
LLMs, originally developed for natural language processing, have proven remarkably versatile. GPT-4, LLaMA, and PaLM are trained on trillions of words, learning not just syntax and semantics but also generalizable reasoning patterns. Researchers soon realized that these models could extend beyond text to other structured data types, including images, tabular data, and time series.

LLMs generalize across domains. Unlike traditional machine learning models that require extensive retraining for each new task, LLMs can leverage few-shot and zero-shot learning to generate insights from raw data with minimal supervision. This ability makes them particularly attractive for time series applications, where labeled data is often scarce and expensive to obtain.

### Applying LLMs for Time Series
By treating numerical data as a language-like sequence, LLMs can perform forecasting, classification, and anomaly detection without requiring domain-specific feature engineering. Instead of building separate models for different time series tasks, a single LLM can handle diverse problems through flexible prompting and adaptation.

However, the transition from NLP to time series is not straightforward. The biggest challenge is the modality gap: LLMs are trained on text, whereas time series data consists of continuous numerical values. Traditional LLM tokenization methods struggle to represent precise floating-point values, making direct application difficult. Additionally, time series often exhibit long-range dependencies, irregular sampling rates, and noise --- characteristics that standard LLM architectures are not inherently designed to handle.

To bridge this gap, researchers have explored several strategies:

- Direct Prompting: Formatting time series as text and feeding it into LLMs without modification.
- Quantization: Converting numerical values into discrete tokens that LLMs can process.
- Alignment Techniques: Training encoders that map time series data into an LLM-compatible embedding space.
- Vision as a Bridge: Transforming time series into visual representations for multimodal models.
- Tool Integration: Using LLMs as a controller for specialized time series processing libraries.

Chronos from AWS and Time-LLM are examples of adapting LLMs for time series analysis.

- Chronos (AWS): Developed to optimize LLM architectures for time series, Chronos introduces discrete tokenization strategies that convert real-valued time series into categorical representations. This allows existing transformer-based LLMs to process time series without modifying their core architecture. AWS has positioned Chronos as an enterprise-ready solution for forecasting, anomaly detection, and time series regression.

[**Forecasting ERCOT Load with Amazon Chronos in Python**\ *Amazon Chronos is a time series LLM. We will use it to predict energy demand in the ERCOT (Electric Reliability Council...*medium.com](https://medium.com/@kylejones_47003/forecasting-ercot-load-with-amazon-chronos-in-python-cf8be7a81a75 "https://medium.com/@kylejones_47003/forecasting-ercot-load-with-amazon-chronos-in-python-cf8be7a81a75")[](https://medium.com/@kylejones_47003/forecasting-ercot-load-with-amazon-chronos-in-python-cf8be7a81a75)
- Time-LLM: A research-driven approach that "reprograms" time series into textual prototypes. Time-LLM allows LLMs like LLaMA-7B to process time series by restructuring the data into a text-like format that retains critical numerical relationships. This method enables LLMs to generate domain-specific prompts, combining natural language reasoning with numerical sequence modeling.

[**TimesFM for Time Series Forecasting in Python using Oil Production Data**\ *TimesFM removes the need for constant model retraining. It generalizes across industries. It scales to massive...*medium.com](https://medium.com/@kylejones_47003/timesfm-for-time-series-forecasting-in-python-using-oil-production-data-b0a59b89d3ff "https://medium.com/@kylejones_47003/timesfm-for-time-series-forecasting-in-python-using-oil-production-data-b0a59b89d3ff")[](https://medium.com/@kylejones_47003/timesfm-for-time-series-forecasting-in-python-using-oil-production-data-b0a59b89d3ff) Beyond these, a range of multimodal LLMs, including BloombergGPT (for financial data), GPT4TS (for general time series processing), and specialized encoders like Auto-TTE (for ECG analysis).

### Bridging the Modality Gap
LLMs have achieved remarkable success in text-based tasks, from language translation to code generation. However, applying them to time series presents a fundamental challenge: the modality gap between natural language and numerical sequences. LLMs are designed to process discrete tokens with structured grammatical rules, while time series data consists of continuous numerical values with complex temporal dependencies. To make LLMs effective for time series, researchers have developed several techniques to bridge this gap.

### Text-Based Training vs. Numerical Time Series
LLMs are trained on vast amounts of text using token-based representations. Each word or subword in a sentence is mapped to a fixed-length vector in a high-dimensional embedding space. Models learn the statistical relationships between tokens to predict the next word, summarize content, or answer questions.

Time series data, in contrast, consists of continuous numerical values, making discrete tokenization difficult. Time series has temporal dependencies and autocorrelation. In the real world, many time series datasets are messy. They can have irregular sampling, missing values, and noise, which LLMs are not designed to handle.

Simply feeding time series values as raw text into an LLM leads to poor numerical comprehension due to the way LLMs tokenize numbers. Standard tokenizers break floating-point values into subwords, often causing inconsistency in number representations. For example, a model trained on text might not recognize "123.45" and "1.2345e2" as equivalent values.

To make time series data compatible with LLMs, researchers have developed five major approaches:

1.  [Direct Prompting --- Formatting time series data as natural language for zero-shot and few-shot learning.]
2.  [Time Series Quantization --- Converting numerical sequences into discrete categorical tokens.]
3.  [Aligning Techniques --- Training encoders to map time series data into an LLM-compatible embedding space.]
4.  [Vision as a Bridge --- Transforming time series into visual representations for multimodal processing.]
5.  [Tool Integration --- Using LLMs to generate code or API calls for external time series models.]

### Direct Prompting: Converting Time Series into LLM-Compatible Formats
Direct prompting treats time series as text. It formats numerical data as natural language. It asks the model to infer patterns. This method requires no fine-tuning. It works with any pre-trained LLM. It allows for zero-shot forecasting and anomaly detection.

Raw time series data does not fit LLMs. A sequence of numbers has no semantic structure (verbs, nouns, objects). Prompting can reformat the data to address this.

Instead of feeding numbers, users describe them. Instead of "72.5, 74.1, 75.3, 76.8," they write, "The temperature has risen steadily for four days. What happens next?"

This forces the model to reason like a human. It predicts outcomes based on learned patterns. It does not perform mathematical calculations. It uses common sense and pattern recognition instead.

Some prompts avoid numbers entirely and focus only on trends. They ask general questions. "Sales have increased each week. Will they continue rising?"

Others keep numerical values but format them carefully. They space out digits. They ensure the model reads numbers consistently. "The last five sales figures were 1 0 5, 1 1 0, 1 1 8, 1 2 5. Predict the next value."

Number-agnostic prompts work for general trends. Number-specific prompts work for structured forecasting. Both have weaknesses. The first lacks precision. The second struggles with arithmetic.

Forecasting works best when trends are clear. If a dataset shows a steady increase, an LLM can extend the pattern. If data fluctuates unpredictably, the model fails. It cannot compute probability distributions. It does not replace statistical forecasting models.

A user asks, "The price of oil has increased each month for a year. What is the likely trend next month?" The model answers, "The price may continue rising based on past data."

This approach works in finance. It works in supply chain forecasting. It provides reasonable guesses based on context. It does not generate precise numeric predictions.

LLMs classify time series patterns without direct computation. They compare input sequences to learned knowledge. A fitness tracker records step counts. A user asks, "Based on these readings, is this person walking or running?" The model recognizes the pattern. It labels the activity correctly.

This works for anomaly detection. A system tracks financial transactions. A user asks, "Do these values suggest fraudulent activity?" The LLM compares past cases. It flags potential risks. It does not calculate statistical probabilities. It finds patterns.

Direct prompting requires no training. It works immediately. It handles qualitative reasoning well. It explains trends in natural language. It assists in exploratory analysis.

It fails at precision. It struggles with complex numerical relationships. It lacks memory. It does not retain long sequences. It provides useful context but cannot replace statistical models.

Prompt engineering will improve and future LLMs will learn better number representation. Until then, businesses can use LLMs for first-pass forecasting. Analysts will refine outputs with traditional models. Prompting will remain a fast, flexible tool for time series interpretation.

Direct prompting works as a starting point. For better accuracy, models need structured numerical input.

### Time Series Quantization
Quantization converts time series into discrete categories and maps continuous values into fixed tokens. This reduces complexity and allows LLMs to process numbers as structured data.

Instead of storing every decimal point, it groups similar numbers. This simplifies processing and improves efficiency.

For example, a stock price of 101.37 could become "A1." A temperature reading of 72.5 could map to "B3." The model no longer handles raw numbers. It recognizes categories instead.

This makes time series more compatible with LLMs. It reduces token count. It eliminates fragmentation in number representation. It improves consistency in predictions.

Vector Quantized Variational Autoencoders (VQ-VAE) learn a dictionary of discrete embeddings that represent numerical values. The model assigns each time series point to the closest vector in a predefined codebook.

For example, a model trained on financial data might learn that:

- Small fluctuations in stock prices map to code A1-A5.
- Larger swings map to codes B1-B5.
- Extreme volatility maps to codes C1-C5.

This approach allows LLMs to interpret numerical patterns as structured text tokens, improving processing efficiency.

Chronos, AWS's LLM-driven time series model, applies VQ-VAE to transform real-valued data into categorical embeddings, making time series forecasting more efficient.

**Raw Data:**

> ***\[123.4, 125.6, 127.8, 126.1, 124.5\]***

**VQ-VAE Encoded Representation:**

> ***\["A1", "A3", "B1", "A4", "A2"\]***

This reduces the number of unique tokens the LLM needs to process while preserving structure.

VQ-VAE requires a pretrained codebook. The model assigns each time series value to a fixed cluster. Instead of handling raw data, it processes learned representations.

Chronos, AWS's time series model, applies this technique. It groups sensor readings into structured tokens. It maps financial data into predefined categories. It reduces noise while preserving key trends.

This approach improves LLM efficiency. It allows models to work with long sequences. It prevents unnecessary tokenization of decimal values.

Another way to convert time series data into discrete tokens is through **K-Means clustering**, which groups similar values into predefined clusters. Each data point is assigned a cluster ID, reducing numerical variation while preserving overall trends.

**Raw Time Series:**

> ***\[0.12, 0.15, 0.87, 0.91, 1.03\]***

**Clustered Representation (5 Clusters):**

> ***\["C1", "C1", "C4", "C4", "C5"\]***

This technique is particularly useful for reducing noise in high-frequency time series such as IoT sensor data. However, it assumes that clusters remain stable over time, which may not always be true.

K-Means works well for structured data. It fails when patterns shift. A sudden spike in stock prices might fall outside predefined clusters. Models must adapt dynamically.

Some systems use fixed bins. They assign ranges instead of clusters. Any number between 0 and 10 maps to "Low." Anything above 90 maps to "High."

This works well for stable datasets. It struggles with extreme values. A rare but critical spike may get misclassified. Fixed bins lose detail. They trade precision for simplicity.

Adaptive binning solves this. It adjusts ranges based on data distribution. It prevents imbalance. It ensures rare but important values get proper representation.

Chronos, for example, combines quantization with deep learning. It trains LLMs on categorical time series. It aligns numerical embeddings with tokenized text. This improves forecasting accuracy. It allows LLMs to handle structured data without losing meaning.

Time-LLM applies a hybrid method. It clusters numerical values. It reprograms time series as text. It allows models to switch between categorical and raw representations.

Hybrid systems improve flexibility. They maintain numerical integrity while optimizing for LLM architectures.

Binning is one of the simplest quantization methods, where continuous values are mapped to predefined categories based on fixed or adaptive intervals.

**Raw Time Series:**

> ***\[0.12, 0.34, 0.68, 0.92\]***

**Binned Representation:**

> ***\["A1", "A2", "A3", "A4"\]***

Fixed bins work well when the range of values is known in advance, but they struggle with skewed distributions where certain ranges occur more frequently.

To handle this, adaptive binning dynamically adjusts bin sizes based on the distribution of data points.


Quantization turns time series into structured sequences. It transforms floating-points into categorical data so the LLMs can process it efficiently.

### Aligning Time Series with LLMs
Time series data consists of continuous values. LLMs work with discrete tokens. This mismatch causes errors. It weakens forecasting and anomaly detection.

Alignment maps time series into a format LLMs understand. It links numerical patterns with textual reasoning. It improves accuracy. It allows models to retain meaning across different data types.

Standard LLMs tokenize numbers like words. They split floating-point values into multiple pieces. This creates inconsistencies. The same number appears as different tokens in different contexts.

Time series models track trends over long sequences. LLMs have short memory. They forget early data points. They miss relationships across time.

Without alignment, LLMs fail at precision. They lose context. They make unreliable predictions.

Alignment forces LLMs to recognize time series as structured data. One method assigns fixed representations to numerical values. Instead of splitting numbers into multiple tokens, the model treats them as single entities.

Another approach uses contrastive learning. The model compares time series patterns with their correct textual descriptions. It learns which sequences match certain outcomes. It refines its predictions by aligning data points with explanations.

Time-LLM uses this strategy. It trains on structured time series and free-text descriptions. It learns to associate stock price movements with economic events. It links medical signals to health outcomes. It predicts future trends based on past observations.

Another method reformats numerical data into natural language. Instead of feeding raw numbers, the model receives structured descriptions. It processes, "The stock price rose steadily for five days before dipping slightly."

This converts numerical reasoning into text-based reasoning. It lets the model work within its strengths. Time-LLM applies this technique to climate forecasting. It turns temperature readings into language. It improves interpretability.

Some models align time series with images. They transform sequences into visual graphs. They feed those graphs into vision-language models. Instead of processing raw numbers, LLMs analyze patterns as visual objects.

This approach works well for ECG signals. It detects heart anomalies without needing structured numerical inputs. It allows LLMs to classify waveforms as normal or irregular.

Finance applies this too. Traders recognize stock patterns visually. LLMs can do the same. They read historical price charts. They classify bullish and bearish trends without direct calculations.

A universal time series model must handle different fields. Medical data differs from industrial sensors. Economic trends follow different patterns than weather cycles. Alignment helps models transfer knowledge across domains.

A model trained on power grid failures learns to detect mechanical breakdowns in factories. A system tracking disease outbreaks adapts to financial crises. Time-LLM uses this approach. It aligns seemingly unrelated time series into common patterns.

Aligned models will track longer sequences. They will store key information and retrieve it when needed. They will integrate text, numbers, and images into a single reasoning system.

### Vision as a Bridge
What if we convert time series data into images?

Time series often appear as plots in research papers and dashboards. Humans recognize patterns in those plots better than in raw numbers. LLMs can do the same when combined with vision models. Instead of reading numbers, they interpret time series as images.

This method relies on vision-language models (VLMs). These models process text and images together. They learn from vast datasets that include charts, graphs, and photos.

Transforming time series into images requires careful design. The most common method plots the data on a graph. A simple line chart works for many tasks. Heatmaps and spectrograms add more detail. Each format highlights different aspects of the data.

A stock price chart shows trends clearly. A spectrogram captures patterns in sound or medical signals. A heatmap reveals correlations in multivariate time series. The right choice depends on the application.

Once converted, the image feeds into a vision model. The model extracts features and passes them to an LLM. This allows reasoning over time series without handling raw numbers.

VLMs combine deep learning with multimodal processing. They use two encoders: one for images, another for text. Models like CLIP, ImageBind, and Flamingo follow this approach. They align visual and textual features in a shared space.

For time series, this means linking patterns in plots to natural language. The model sees a stock price chart and describes it. It recognizes trends, anomalies, and patterns. A user can ask questions like, "Does this chart show an upward trend?" The model answers based on what it sees.

This method works well for exploratory analysis. It helps with pattern recognition and anomaly detection. It does not require complex numerical calculations. Instead, it relies on the model's learned ability to recognize shapes and trends.

Financial analysts use stock charts to spot trends. Vision models can do the same. By training on historical data, they learn to classify price movements. They recognize patterns like head-and-shoulders formations or breakouts.

In healthcare, doctors analyze ECG and EEG signals as plots. Vision-based models automate this process. They detect irregular heartbeats and brain wave anomalies. They assist in diagnosis by highlighting abnormal patterns.

Energy and utilities rely on sensor data. Instead of processing raw values, a model can analyze waveform plots. It detects faults in machinery or predicts failures. This improves maintenance planning and reduces downtime.

This method leverages LLMs without requiring them to process raw numbers. It taps into the strength of vision models. It allows for reasoning over time series in a natural way.

The main weakness is precision. Vision-based models recognize broad trends but lack fine-grained accuracy. They do not perform exact calculations. They work best for qualitative insights, not precise forecasting.

Another challenge is scalability. Converting large datasets into images takes time. Processing many images requires powerful hardware. This limits real-time applications.

Despite these drawbacks, the method has proven useful. It expands what LLMs can do with time series. It allows for deeper integration of structured and unstructured data.

Future models will refine this approach. They will combine LLMs with more advanced vision encoders. They will integrate textual prompts with visual analysis more seamlessly.

Some will use vision for pattern recognition and specialized models for numerical calculations. Others will combine different representations, blending images, text, and structured data.

### Tool Integration
LLMs can be used not just for direct inference but also as controllers for external time series models. Instead of processing time series directly, LLMs generate code, API calls, or structured queries to retrieve results from specialized forecasting engines.

Examples include:

- ToolLLM, which generates API calls for external time series forecasting models.
- Auto-TTE, which creates interpretable ECG classifications by translating time series data into LLM-executable scripts.

This approach leverages the strengths of both LLMs and traditional time series models, but it requires external dependencies and structured API interactions.

### Comparative Evaluation of Different Strategies
Each method for bridging the modality gap has trade-offs:


You don't have to choose just one approach. You can do an ensemble of these.

### LLMs in Time Series Analysis
LLMs are better able to handle time series data than they were a few years ago. But there are still issues around context windows for large datasets, numerical precision and tokenization, and scalability to apply LLMs to high frequency data like real-time financial or IoT data.
