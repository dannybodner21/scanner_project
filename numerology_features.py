#!/usr/bin/env python3
"""
FOCUSED numerology features for crypto ML models
Only 10 most powerful numerology features to keep total under 50
"""

import re
from datetime import datetime
from typing import Dict, Any
import pandas as pd

# Pythagorean letter mapping (A=1, B=2, ..., Z=26, then mod 9)
LETTER_MAP = {c: ((i % 9) or 9) for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ", start=1)}

def digital_root(n: int) -> int:
    """Calculate digital root of a number"""
    if n == 0: 
        return 0
    return 9 if n % 9 == 0 else n % 9

def sum_digits_str(s: str) -> int:
    """Sum all digits in a string"""
    return sum(int(ch) for ch in re.findall(r"\d", s))

def pyth_sum(s: str) -> int:
    """Calculate Pythagorean sum of letters in a string"""
    s = re.sub(r'[^A-Za-z]', '', s).upper()
    return sum(LETTER_MAP.get(ch, 0) for ch in s)

def master_pre_reduction(total: int) -> int:
    """Check if number is a master number (11, 22, 33) before final reduction"""
    return total if total in (11, 22, 33) else 0

def get_numerology_features(timestamp: datetime, symbol: str) -> Dict[str, Any]:
    """Get ONLY the most powerful numerology features - 10 total"""
    
    # Date numerology
    ymd = f"{timestamp.year:04d}{timestamp.month:02d}{timestamp.day:02d}"
    hm = f"{timestamp.hour:02d}{timestamp.minute:02d}"
    date_sum = sum_digits_str(ymd)
    time_sum = sum_digits_str(hm)
    
    # Symbol numerology
    m = re.search(r'([A-Z]{2,6})[/:\-]?([A-Z]{2,6})$', symbol.upper())
    base, quote = (m.group(1), m.group(2)) if m else (symbol.upper(), "")
    b_sum, q_sum = pyth_sum(base), pyth_sum(quote)
    
    # ONLY the most powerful numerology features (10 total):
    all_features = {
        # Core roots (3 features)
        "date_root": digital_root(date_sum),
        "time_root": digital_root(time_sum), 
        "symbol_root": digital_root(pyth_sum(symbol)),
        
        # Master number flags (3 features)
        "master_flag_date": int(master_pre_reduction(date_sum) in (11, 22, 33)),
        "master_flag_time": int(master_pre_reduction(time_sum) in (11, 22, 33)),
        "master_symbol_flag": int(any(master_pre_reduction(x) in (11, 22, 33) for x in (b_sum, q_sum))),
        
        # Pattern detection (2 features)
        "repeating_digits_date": int(len(ymd) != len(set(ymd))),
        "palindrome_flag_date": int(ymd == ymd[::-1]),
        
        # Master number density (1 feature)
        "master_density": sum([
            int(master_pre_reduction(date_sum) in (11, 22, 33)),
            int(master_pre_reduction(time_sum) in (11, 22, 33)),
            int(any(master_pre_reduction(x) in (11, 22, 33) for x in (b_sum, q_sum)))
        ]),
        
        # Root harmony (1 feature)
        "date_time_harmony": int(digital_root(date_sum) == digital_root(time_sum))
    }
    
    return all_features

def add_numerology_features(df, symbol: str = "XRPUSDT"):
    """
    Add numerology features to a DataFrame with timestamp column
    """
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must have 'timestamp' column")
    
    # Convert timestamp to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    # Apply numerology features to each row
    numerology_features = []
    for _, row in df.iterrows():
        features = get_numerology_features(row['timestamp'], symbol)
        numerology_features.append(features)
    
    # Convert to DataFrame and merge
    numerology_df = pd.DataFrame(numerology_features, index=df.index)
    
    # Add to original DataFrame
    result_df = pd.concat([df, numerology_df], axis=1)
    
    return result_df