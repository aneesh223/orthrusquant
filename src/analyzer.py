import pandas as pd
import numpy as np
from functools import lru_cache
import re
import warnings
import logging

warnings.filterwarnings("ignore", category=FutureWarning)

# Global caches
_sentiment_cache = {}
_batch_cache = {}
_cache_hits = 0
_cache_misses = 0

# VADER sentiment analysis imports
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader_analyzer = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("VADER not available, using fallback sentiment analysis")

# DistilRoBERTa model imports - lazy loaded for better performance
_ai_model = None
_ai_tokenizer = None

def _ensure_model():
    """Lazy load DistilRoBERTa model and tokenizer"""
    global _ai_model, _ai_tokenizer
    if _ai_model is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            logging.info("Loading DistilRoBERTa financial sentiment model...")
            model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
            _ai_tokenizer = AutoTokenizer.from_pretrained(model_name)
            _ai_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            if torch.cuda.is_available():
                device = torch.device("cuda")
                _ai_model = _ai_model.to(device)
                logging.info(f"DistilRoBERTa loaded on GPU: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
                _ai_model = _ai_model.to(device)
                logging.info("DistilRoBERTa loaded on Apple Silicon GPU (MPS)")
            else:
                device = torch.device("cpu")
                logging.info("DistilRoBERTa loaded on CPU")
            
            _ai_model.eval()
            
        except Exception as e:
            logging.error(f"Error loading DistilRoBERTa model: {e}")
            logging.info("Falling back to VADER-only analysis...")
            return False
    return True

# Pre-compiled regex patterns
_SOURCE_TAG_PATTERNS = [
    re.compile(r'^\[.*?\]\s*'),      # Remove from start
    re.compile(r'\s*\[.*?\]$'),      # Remove from end  
    re.compile(r'\s*\[.*?\]\s*')     # Remove from middle (with proper spacing)
]

# Enhanced Financial Lexicon for VADER
FINANCIAL_LEXICON = {
    'error': 0.0, 'loom': 0.0, 'vice': 0.0, 'gross': 0.0, 'mine': 0.0, 'arrest': 0.0, 'fool': 0.0, 'motley': 0.0,
    'tank': -2.5, 'plunge': -3.0, 'brutal': -3.0, 'crash': -3.0, 'collapse': -3.0,
    'miss': -2.5, 'fall short': -2.5, 'falls short': -2.5, 'fell short': -2.5,
    'disappointing': -2.0, 'decline': -2.0, 'drop': -2.0, 'tumble': -2.5,
    'slump': -2.5, 'weak': -1.5, 'poor': -2.0, 'loss': -2.0, 'deficit': -2.0,
    'flat': -1.5, 'stagnant': -1.5, 'choppy': -1.0, 'sideways': -1.0,
    'beat': 2.5, 'crush': 3.0, 'soar': 3.0, 'skyrocket': 3.5, 'rally': 2.5,
    'exceed': 2.0, 'outperform': 2.0, 'strong': 2.0, 'robust': 2.0,
    'growth': 1.5, 'gains': 2.0, 'profit': 2.0, 'upgrade': 2.0, 'bullish': 2.5,
    'surprise': 1.5, 'revenue': 1.0, 'earnings': 1.0, 'call': 1.0, 'shorting': -2.0
}

def _update_vader_lexicon():
    """Update VADER's lexicon with financial terms"""
    if VADER_AVAILABLE:
        _vader_analyzer.lexicon.update(FINANCIAL_LEXICON)
        logging.info("VADER enhanced with financial lexicon")

# Initialize enhanced VADER
if VADER_AVAILABLE:
    _update_vader_lexicon()

# Source categorization using numpy arrays
_PRIMARY_SOURCES_ARRAY = np.array(['SEC EDGAR', 'SEC', 'U.S. Securities and Exchange Commission', 
                                  'Company Investor Relations', 'Investor Relations'])
_INSTITUTIONAL_SOURCES_ARRAY = np.array(['Bloomberg', 'Reuters', 'The Wall Street Journal', 'Financial Times', 
                                        'Barron\'s', 'Morningstar', 'Investor\'s Business Daily'])
_DATA_AGGREGATORS_ARRAY = np.array(['Yahoo Finance', 'MarketWatch', 'Seeking Alpha', 'Zacks', 'TipRanks', 'Benzinga'])
_ENTERTAINMENT_SOURCES_ARRAY = np.array(['The Motley Fool', 'CNBC', 'CNN Business', 'Fox Business', 'MSN',
                                        'USA Today', 'Forbes', 'Fortune', 'Business Insider', 'Reddit',
                                        'Twitter', 'StockTwits', 'Robinhood Snacks', 'TheStreet'])

# Convert to frozensets for O(1) lookup
PRIMARY_SOURCES = frozenset(_PRIMARY_SOURCES_ARRAY)
INSTITUTIONAL_SOURCES = frozenset(_INSTITUTIONAL_SOURCES_ARRAY)
DATA_AGGREGATORS = frozenset(_DATA_AGGREGATORS_ARRAY)
ENTERTAINMENT_SOURCES = frozenset(_ENTERTAINMENT_SOURCES_ARRAY)

# Source weights using numpy for faster lookups
_SOURCE_WEIGHT_KEYS = np.array(list(PRIMARY_SOURCES) + list(INSTITUTIONAL_SOURCES) + list(DATA_AGGREGATORS) + list(ENTERTAINMENT_SOURCES))
_SOURCE_WEIGHT_VALUES = np.array([2.0] * len(PRIMARY_SOURCES) + 
                                [1.8, 1.8, 1.8, 1.8, 1.7, 1.6, 1.6] +  # Institutional weights
                                [1.2, 1.2, 1.0, 1.0, 1.0, 1.5] +        # Aggregator weights (Benzinga=1.5)
                                [0.6, 0.8, 0.7, 0.7, 0.6, 0.5, 0.7, 0.7, 0.6, 0.4, 0.3, 0.4, 0.3, 0.5])  # Entertainment

SOURCE_WEIGHTS = dict(zip(_SOURCE_WEIGHT_KEYS, _SOURCE_WEIGHT_VALUES))

# Cache hybrid analyzer instances
@lru_cache(maxsize=1)
def get_hybrid_analyzers():
    """Get cached VADER and DistilRoBERTa analyzers"""
    vader_analyzer = get_vader_analyzer()
    ai_model, ai_tokenizer = get_ai_analyzer()
    return vader_analyzer, ai_model, ai_tokenizer

@lru_cache(maxsize=1)
def get_vader_analyzer():
    """Get cached VADER analyzer"""
    if VADER_AVAILABLE:
        return _vader_analyzer
    return None

@lru_cache(maxsize=1)
def get_ai_analyzer():
    """Get cached DistilRoBERTa analyzer with lazy initialization"""
    if _ensure_model():
        return _ai_model, _ai_tokenizer
    return None, None

def get_hybrid_sentiment(text, ticker=""):
    """Hybrid sentiment analysis: DistilRoBERTa for accuracy + VADER for nuanced scoring"""
    global _cache_hits, _cache_misses
    
    # Check global cache first for ultra-fast lookup - include ticker in cache key
    cache_key = hash(f"{ticker}_{text}")
    if cache_key in _sentiment_cache:
        _cache_hits += 1
        return _sentiment_cache[cache_key]
    
    _cache_misses += 1
    
    vader_analyzer, ai_model, ai_tokenizer = get_hybrid_analyzers()
    
    # Clean text for analysis
    cleaned_text = clean_headline(text)
    
    # Get VADER score (nuanced -1 to 1 range)
    vader_score = 0.0
    if vader_analyzer is not None:
        try:
            vader_scores = vader_analyzer.polarity_scores(cleaned_text)
            vader_score = vader_scores['compound']
        except Exception as e:
            print(f"VADER error: {e}")
            vader_score = get_simple_sentiment(text)
    else:
        vader_score = get_simple_sentiment(text)
    
    # Get DistilRoBERTa classification (for accuracy)
    ai_classification = None
    ai_confidence = 0.5
    
    if ai_model is not None and ai_tokenizer is not None:
        try:
            import torch
            
            # Tokenize and get model prediction
            inputs = ai_tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
                
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = ai_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # DistilRoBERTa outputs: [Negative, Neutral, Positive]
            negative_score = predictions[0][0].item()
            neutral_score = predictions[0][1].item()
            positive_score = predictions[0][2].item()
            
            # Determine classification and confidence
            max_score = max(negative_score, neutral_score, positive_score)
            ai_confidence = max_score
            
            if positive_score == max_score:
                ai_classification = 'positive'
            elif negative_score == max_score:
                ai_classification = 'negative'
            else:
                ai_classification = 'neutral'
                
        except Exception as e:
            print(f"DistilRoBERTa error: {e}")
    
    # Hybrid scoring logic
    if ai_classification is None:
        # Fallback to pure VADER
        result = vader_score
    else:
        # Use DistilRoBERTa classification to validate/correct VADER direction
        if ai_classification == 'positive':
            # If DistilRoBERTa says positive, ensure VADER score is positive
            if vader_score < 0:
                # VADER disagrees - use moderate positive score weighted by confidence
                result = abs(vader_score) * ai_confidence * 0.7
            else:
                # VADER agrees - use VADER's nuanced score
                result = vader_score
        
        elif ai_classification == 'negative':
            # If DistilRoBERTa says negative, ensure VADER score is negative
            if vader_score > 0:
                # VADER disagrees - use moderate negative score weighted by confidence
                result = -abs(vader_score) * ai_confidence * 0.7
            else:
                # VADER agrees - use VADER's nuanced score
                result = vader_score
        
        else:  # neutral
            # DistilRoBERTa says neutral - dampen VADER score
            if ai_confidence > 0.6:
                # High confidence neutral - strongly dampen
                result = vader_score * 0.2
            else:
                # Low confidence neutral - moderately dampen
                result = vader_score * 0.5
    
    # Cache the result for ultra-fast future lookups
    _sentiment_cache[cache_key] = result
    
    return result

def get_hybrid_sentiment_batch(texts, ticker="", batch_size=32):
    """Batch processing with caching and memory management"""
    global _cache_hits, _cache_misses, _batch_cache
    
    vader_analyzer, ai_model, ai_tokenizer = get_hybrid_analyzers()
    
    if ai_model is None or ai_tokenizer is None:
        return get_vader_sentiment_batch(texts)
    
    # Check batch cache
    batch_cache_key = hash((ticker, tuple(texts)))
    if batch_cache_key in _batch_cache:
        _cache_hits += len(texts)
        return _batch_cache[batch_cache_key]
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
            ai_model = ai_model.to(device)
            logging.info(f"Using GPU acceleration: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            ai_model = ai_model.to(device)
            logging.info("Using Apple Silicon GPU acceleration (MPS)")
        else:
            device = torch.device("cpu")
            logging.info("Using CPU processing")
        
        results = []
        
        # VADER processing
        vader_scores = []
        if vader_analyzer is not None:
            cleaned_texts = [clean_headline(text) for text in texts]
            
            for cleaned_text in cleaned_texts:
                try:
                    scores = vader_analyzer.polarity_scores(cleaned_text)
                    vader_scores.append(scores['compound'])
                except:
                    vader_scores.append(get_simple_sentiment(cleaned_text))
        else:
            vader_scores = [get_simple_sentiment(text) for text in texts]
        
        # DistilRoBERTa processing
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_vader_scores = vader_scores[i:i + batch_size]
            
            cleaned_batch = [clean_headline(text) for text in batch_texts]
            
            inputs = ai_tokenizer(cleaned_batch, return_tensors="pt", truncation=True, 
                                 padding=True, max_length=256)
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                if device.type == "cuda" and hasattr(torch, 'autocast'):
                    with torch.autocast(device_type='cuda'):
                        outputs = ai_model(**inputs)
                        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                else:
                    outputs = ai_model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Vectorized hybrid logic
            predictions_np = predictions.cpu().numpy()
            
            max_indices = np.argmax(predictions_np, axis=1)
            max_confidences = np.max(predictions_np, axis=1)
            
            for j in range(len(batch_texts)):
                vader_score = batch_vader_scores[j]
                ai_confidence = max_confidences[j]
                ai_class_idx = max_indices[j]
                
                # DistilRoBERTa outputs: [Negative, Neutral, Positive]
                if ai_class_idx == 2:  # Positive
                    if vader_score < 0:
                        hybrid_score = abs(vader_score) * ai_confidence * 0.7
                    else:
                        hybrid_score = vader_score
                elif ai_class_idx == 0:  # Negative
                    if vader_score > 0:
                        hybrid_score = -abs(vader_score) * ai_confidence * 0.7
                    else:
                        hybrid_score = vader_score
                else:  # Neutral
                    if ai_confidence > 0.6:
                        hybrid_score = vader_score * 0.2
                    else:
                        hybrid_score = vader_score * 0.5
                
                results.append(hybrid_score)
            
            # Removed torch.cuda.empty_cache() to prevent allocator thrashing
        
        _batch_cache[batch_cache_key] = results
        _cache_misses += len(texts)
        
        return results
        
    except Exception as e:
        logging.error(f"Hybrid batch error: {e}")
        return get_vader_sentiment_batch(texts)

@lru_cache(maxsize=1000)
def remove_source_tags(headline):
    """Remove source tags like [benzinga], [reuters] from headlines"""
    if not headline:
        return headline
    
    cleaned = headline
    # Apply patterns in order: start, end, then middle
    cleaned = _SOURCE_TAG_PATTERNS[0].sub('', cleaned)  # Remove from start
    cleaned = _SOURCE_TAG_PATTERNS[1].sub('', cleaned)  # Remove from end
    cleaned = _SOURCE_TAG_PATTERNS[2].sub(' ', cleaned)  # Remove from middle, replace with space
    
    # Clean up multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.strip()

def get_vader_sentiment_batch(texts, batch_size=None):
    """Batch process multiple texts using VADER only"""
    analyzer = get_vader_analyzer()
    
    if analyzer is None:
        return [get_simple_sentiment(text) for text in texts]
    
    try:
        results = []
        
        for text in texts:
            cleaned_text = clean_headline(text)
            scores = analyzer.polarity_scores(cleaned_text)
            results.append(scores['compound'])
        
        return results
        
    except Exception as e:
        logging.error(f"VADER batch error: {e}")
        return [get_simple_sentiment(text) for text in texts]

@lru_cache(maxsize=1000)
def get_simple_sentiment(text):
    """Simple rule-based sentiment analysis as fallback"""
    text_lower = text.lower()
    
    positive_words = ['good', 'great', 'excellent', 'strong', 'beat', 'exceed', 'rally', 'soar', 'gain', 'profit', 'bullish', 'upgrade']
    negative_words = ['bad', 'poor', 'weak', 'miss', 'fall', 'drop', 'decline', 'loss', 'bearish', 'downgrade', 'crash', 'plunge']
    
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return min(0.8, positive_count * 0.2)
    elif negative_count > positive_count:
        return max(-0.8, -negative_count * 0.2)
    else:
        return 0.0

@lru_cache(maxsize=1000)
def clean_headline(text):
    """Headline cleaning with financial phrase replacements"""
    if not text:
        return text
        
    text_lower = text.lower()
    
    # Financial phrase replacements for better VADER analysis
    replacements = {
        'earnings fall short': 'disappointing earnings',
        'falls short': 'disappointing', 'fell short': 'disappointing',
        'revenue tops estimates': 'strong revenue', 'tops estimates': 'beat expectations',
        'exceeds expectations': 'beat expectations', 'below estimates': 'miss expectations',
        'break losing streak': 'rally', 'snaps losing streak': 'rally',
        'recovers from': 'rally', 'bounce back': 'rally',
        'top pick': 'best stock', 'buy rating': 'strong buy', 'strong buy': 'excellent',
        'outperform': 'winner', 'stock to buy': 'good investment',
        
        # Context-aware fixes: Handle contrarian/bullish statements
        'saying it would miss': 'predicting decline but wrong',  # Cramer-style contrarian
        'would miss': 'expected to decline',
        'has it wrong': 'incorrect bearish prediction',
        'wall street has it wrong': 'incorrect bearish prediction',
        'numbers are too low': 'estimates too conservative',
        'estimates too low': 'conservative estimates',
        'was trashing': 'unfairly criticized',
        'trashing': 'criticizing unfairly'
    }
    
    for phrase, replacement in replacements.items():
        if phrase in text_lower:
            text_lower = text_lower.replace(phrase, replacement)
    
    return text_lower

@lru_cache(maxsize=1000)
def validate_target(headline, ticker, raw_score):
    """Target validation with caching"""
    headline_lower = headline.lower()
    ticker_lower = ticker.lower()
    
    if " for " in headline_lower:
        parts = headline_lower.split(" for ", 1)
        if len(parts) > 1:
            target = parts[1].split()[0].strip(".,;:!?")
            
            generics = frozenset(['investors', 'shareholders', 'holders', 'stock', 'market', 
                                'trading', 'tech', 'semis', 'sector', 'growth', 'economy'])
            
            if target != ticker_lower and target not in generics:
                return 0.0
    
    return raw_score

@lru_cache(maxsize=500)
def categorize_source(source):
    """Source categorization with case-insensitive matching"""
    # Convert to title case for consistent matching
    source_title = source.title()
    
    if source_title in PRIMARY_SOURCES:
        return 'primary'
    elif source_title in INSTITUTIONAL_SOURCES:
        return 'institutional'
    elif source_title in DATA_AGGREGATORS:
        return 'aggregator'
    elif source_title in ENTERTAINMENT_SOURCES:
        return 'entertainment'
    else:
        # Fast keyword matching (case-insensitive)
        source_lower = source.lower()
        if any(kw in source_lower for kw in ('sec', 'securities', 'investor relations')):
            return 'primary'
        elif any(kw in source_lower for kw in ('bloomberg', 'reuters', 'wall street', 'financial times')):
            return 'institutional'
        elif any(kw in source_lower for kw in ('yahoo finance', 'marketwatch', 'seeking alpha', 'benzinga')):
            return 'aggregator'
        else:
            return 'entertainment'

def get_sentiment(parsed_data, ticker, timeframe_days=30):
    """Sentiment analysis with hybrid VADER + DistilRoBERTa approach and caching"""
    global _cache_hits, _cache_misses
    
    if not parsed_data:
        return pd.DataFrame()

    # Create DataFrame with optimized dtypes
    df = pd.DataFrame(parsed_data, columns=['Timestamp', 'Headline', 'Source'])
    
    if df.empty:
        return pd.DataFrame()
    
    # Convert timestamps to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)
    
    # Optimize memory usage by using categorical data types
    df['Source'] = df['Source'].astype('category')
    
    # Get cached hybrid analyzers
    vader_analyzer, ai_model, ai_tokenizer = get_hybrid_analyzers()
    
    # Vectorized headline cleaning with caching
    df['Cleaned_Headline'] = df['Headline'].apply(clean_headline)
    
    logging.info(f"Analyzing {len(df)} news articles... (Cache hits: {_cache_hits}, misses: {_cache_misses})")
    
    # Add progress bar for sentiment analysis
    try:
        from tqdm import tqdm
        use_progress_bar = True
    except ImportError:
        use_progress_bar = False
    
    # Hybrid sentiment analysis with larger batches
    if ai_model is not None and ai_tokenizer is not None and vader_analyzer is not None:
        logging.info("Using hybrid VADER + DistilRoBERTa analysis...")
        
        headlines_list = df['Cleaned_Headline'].tolist()
        
        if use_progress_bar:
            logging.info("Processing headlines with hybrid approach...")
            raw_scores = get_hybrid_sentiment_batch(headlines_list, ticker, batch_size=64)
        else:
            raw_scores = get_hybrid_sentiment_batch(headlines_list, ticker, batch_size=64)
            
        raw_scores = np.array(raw_scores, dtype=np.float32)
    elif vader_analyzer is not None:
        logging.info("Using VADER with enhanced financial lexicon...")
        headlines_list = df['Cleaned_Headline'].tolist()
        raw_scores = get_vader_sentiment_batch(headlines_list)
        raw_scores = np.array(raw_scores, dtype=np.float32)
    else:
        logging.info("Using fallback sentiment analysis...")
        if use_progress_bar:
            raw_scores = np.array([get_simple_sentiment(headline) 
                                 for headline in tqdm(df['Cleaned_Headline'], desc="Processing headlines")], dtype=np.float32)
        else:
            raw_scores = np.array([get_simple_sentiment(headline) for headline in df['Cleaned_Headline']], dtype=np.float32)
    
    df['Raw_Score'] = raw_scores
    
    # Vectorized target validation using numpy operations
    headlines_array = df['Headline'].values
    ticker_lower = ticker.lower()
    
    valid_scores = np.zeros(len(headlines_array), dtype=np.float32)
    
    for i, (headline, raw_score) in enumerate(zip(headlines_array, raw_scores)):
        headline_lower = headline.lower()
        
        if " for " in headline_lower:
            parts = headline_lower.split(" for ", 1)
            if len(parts) > 1:
                target = parts[1].split()[0].strip(".,;:!?")
                
                generics = {'investors', 'shareholders', 'holders', 'stock', 'market', 
                           'trading', 'tech', 'semis', 'sector', 'growth', 'economy'}
                
                if target != ticker_lower and target not in generics:
                    valid_scores[i] = 0.0
                else:
                    valid_scores[i] = raw_score
            else:
                valid_scores[i] = raw_score
        else:
            valid_scores[i] = raw_score
    
    df['Raw_Score'] = valid_scores
    
    # Source processing with vectorized operations
    source_categories = df['Source'].apply(categorize_source).astype('category')
    df['Source_Type'] = source_categories
    
    # Vectorized weight calculation
    source_weights = np.array([SOURCE_WEIGHTS.get(s.title(), SOURCE_WEIGHTS.get(s, 1.0)) 
                              for s in df['Source']], dtype=np.float32)
    df['Weight'] = source_weights
    df['Compound_Score'] = np.clip(df['Raw_Score'] * df['Weight'], -1.0, 1.0).astype('float32')
    
    # Time grouping
    if timeframe_days <= 1:
        df['TimeGroup'] = df['Timestamp'].dt.floor('h')
    else:
        df['TimeGroup'] = df['Timestamp'].dt.date
    
    # Aggregation using numpy operations
    source_masks = {
        'primary': (source_categories == 'primary').values,
        'institutional': (source_categories == 'institutional').values, 
        'aggregator': (source_categories == 'aggregator').values,
        'entertainment': (source_categories == 'entertainment').values
    }
    
    def weighted_mean(group):
        weights = group['Weight'].values.astype(np.float32)
        scores = group['Compound_Score'].values.astype(np.float32)
        weight_sum = np.sum(weights)
        return np.sum(scores * weights) / weight_sum if weight_sum > 0 else 0.0
    
    # Main aggregation
    result_df = df.groupby('TimeGroup', sort=False).agg({
        'Raw_Score': 'mean', 
        'Weight': 'mean'
    }).astype(np.float32)
    
    # Calculate weighted compound scores
    try:
        weighted_scores = df.groupby('TimeGroup', sort=False).apply(weighted_mean, include_groups=False)
    except TypeError:
        weighted_scores = df.groupby('TimeGroup', sort=False).apply(weighted_mean)
    
    result_df['Compound_Score'] = weighted_scores.astype(np.float32)
    
    # Sentiment and count calculations using pre-computed masks
    for source_type, mask in source_masks.items():
        if np.any(mask):
            source_df = df[mask]
            sentiment_col = f'{source_type.title()}_Sentiment'
            count_col = f'{source_type.title()}_Count'
            
            try:
                sentiment_scores = source_df.groupby('TimeGroup', sort=False).apply(weighted_mean, include_groups=False)
            except TypeError:
                sentiment_scores = source_df.groupby('TimeGroup', sort=False).apply(weighted_mean)
            
            counts = source_df.groupby('TimeGroup', sort=False).size()
            
            result_df[sentiment_col] = sentiment_scores.reindex(result_df.index, fill_value=0).astype(np.float32)
            result_df[count_col] = counts.reindex(result_df.index, fill_value=0).astype(np.int16)
        else:
            result_df[f'{source_type.title()}_Sentiment'] = np.float32(0)
            result_df[f'{source_type.title()}_Count'] = np.int16(0)
    
    # Memory cleanup
    del df, headlines_array, raw_scores, valid_scores, source_weights, source_categories
    
    # Print cache statistics
    hit_rate = _cache_hits / (_cache_hits + _cache_misses) * 100 if (_cache_hits + _cache_misses) > 0 else 0
    logging.info(f"Analysis complete with {hit_rate:.1f}% cache hit rate")
    
    return result_df

@lru_cache(maxsize=1000)
def get_top_headlines(parsed_data_tuple, ticker, timeframe_days=30):
    """Top headlines extraction with hybrid sentiment analysis"""
    if not parsed_data_tuple:
        return "N/A", "N/A"

    # Convert tuple back to list for DataFrame creation
    parsed_data = list(parsed_data_tuple)
    df = pd.DataFrame(parsed_data, columns=['Timestamp', 'Headline', 'Source'])
    
    if df.empty:
        return "No significant news.", "No significant news."
    
    # Get cached hybrid analyzers
    vader_analyzer, ai_model, ai_tokenizer = get_hybrid_analyzers()
    
    # Vectorized operations - use clean_headline function
    df['Cleaned'] = df['Headline'].apply(clean_headline)
    
    if ai_model is not None and ai_tokenizer is not None and vader_analyzer is not None:
        # Use hybrid batch processing for better performance
        cleaned_headlines = df['Cleaned'].tolist()
        scores = get_hybrid_sentiment_batch(cleaned_headlines, ticker, batch_size=16)
        df['Score'] = scores
    elif vader_analyzer is not None:
        # Use VADER only
        cleaned_headlines = df['Cleaned'].tolist()
        scores = get_vader_sentiment_batch(cleaned_headlines)
        df['Score'] = scores
    else:
        # Fallback to simple sentiment
        df['Score'] = df['Cleaned'].apply(get_simple_sentiment)
        
    df['Score'] = df.apply(lambda row: validate_target(row['Headline'], ticker, row['Score']), axis=1)

    # Sort once for efficiency
    df = df.sort_values(by='Score', ascending=False)
    
    top_story = df.iloc[0]
    bottom_story = df.iloc[-1]
    
    if top_story['Score'] > 0.1:
        # Remove source tags like [benzinga] from headlines
        cleaned_headline = remove_source_tags(top_story['Headline'])
        best_headline = f"{cleaned_headline} (Score: {top_story['Score']:.2f})"
    else:
        best_headline = "No significant positive news found."

    if bottom_story['Score'] < -0.1:
        # Remove source tags like [benzinga] from headlines
        cleaned_headline = remove_source_tags(bottom_story['Headline'])
        worst_headline = f"{cleaned_headline} (Score: {bottom_story['Score']:.2f})"
    else:
        worst_headline = "No significant negative news found."
    
    return best_headline, worst_headline

# Wrapper function to handle the tuple conversion for caching
def get_top_headlines_wrapper(parsed_data, ticker, timeframe_days=30):
    """Wrapper to convert list to tuple for caching"""
    if not parsed_data:
        return "N/A", "N/A"
    
    # Convert to tuple for hashing (required for caching)
    parsed_data_tuple = tuple(tuple(row) if isinstance(row, list) else row for row in parsed_data)
    return get_top_headlines(parsed_data_tuple, ticker, timeframe_days)

def clear_sentiment_caches():
    """Clear all sentiment analysis caches and report statistics"""
    global _sentiment_cache, _batch_cache, _cache_hits, _cache_misses
    
    # Report final statistics
    total_requests = _cache_hits + _cache_misses
    hit_rate = (_cache_hits / total_requests * 100) if total_requests > 0 else 0
    
    logging.info(f"Clearing sentiment caches...")
    logging.info(f"   Cache Statistics:")
    logging.info(f"   - Total requests: {total_requests:,}")
    logging.info(f"   - Cache hits: {_cache_hits:,}")
    logging.info(f"   - Cache misses: {_cache_misses:,}")
    logging.info(f"   - Hit rate: {hit_rate:.1f}%")
    logging.info(f"   - Individual cache size: {len(_sentiment_cache):,}")
    logging.info(f"   - Batch cache size: {len(_batch_cache):,}")
    
    # Clear caches
    _sentiment_cache.clear()
    _batch_cache.clear()
    _cache_hits = 0
    _cache_misses = 0
    
    # Clear LRU caches
    clean_headline.cache_clear()
    validate_target.cache_clear()
    categorize_source.cache_clear()
    get_top_headlines.cache_clear()
    
    logging.info("All caches cleared")

def get_cache_stats():
    """Get current cache statistics"""
    global _cache_hits, _cache_misses, _sentiment_cache, _batch_cache
    
    total_requests = _cache_hits + _cache_misses
    hit_rate = (_cache_hits / total_requests * 100) if total_requests > 0 else 0
    
    return {
        'total_requests': total_requests,
        'cache_hits': _cache_hits,
        'cache_misses': _cache_misses,
        'hit_rate': hit_rate,
        'individual_cache_size': len(_sentiment_cache),
        'batch_cache_size': len(_batch_cache)
    }