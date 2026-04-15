import pandas as pd

df = pd.read_csv(r'phrase_reduction_v2/image_compiled_phrases.csv')
st = pd.read_csv(r'phrase_reduction_v2/phrase_shortlist.csv')

# Subtopics system prompt (sent once per call with --no-caching, or cached)
subtopic_lines = [f"{row['SubTopic']}: {row['Description']}" for _, row in st.iterrows()]
subtopic_text = '\n'.join(subtopic_lines)
print(f'Subtopics block chars: {len(subtopic_text)}  (~{len(subtopic_text)//4} tokens)')

# Fixed instruction overhead (task description + JSON schema example) ~300 tokens
FIXED_OVERHEAD_TOKENS = 500  # system prompt + task instructions + JSON schema

# Per-image input
df['input_text'] = df['rawUserComments'].fillna('') + ' ' + df['humanCuratedPhrases'].fillna('')
char_stats = df['input_text'].str.len()
print(f'\nPer-image input text (comments + phrases):')
print(f'  Min:  {char_stats.min()} chars  (~{char_stats.min()//4} tokens)')
print(f'  Mean: {char_stats.mean():.0f} chars  (~{char_stats.mean()/4:.0f} tokens)')
print(f'  Max:  {char_stats.max()} chars  (~{char_stats.max()//4} tokens)')

# Typical per-call token breakdown
# Input = subtopics (~350) + fixed instructions (~500) + image text (~mean)
mean_input = len(subtopic_text)//4 + FIXED_OVERHEAD_TOKENS + int(char_stats.mean()/4)
# Output = JSON extraction, ~250 tokens per image
mean_output = 250

print(f'\nPer-call token estimate:')
print(f'  Input tokens:  ~{mean_input}')
print(f'  Output tokens: ~{mean_output}')

# Claude Sonnet 4 pricing (as of 2025): $3/M input, $15/M output
input_price_per_M = 3.00
output_price_per_M = 15.00

cost_per_call = (mean_input / 1_000_000 * input_price_per_M) + (mean_output / 1_000_000 * output_price_per_M)
print(f'\nCost per image (Sonnet 4):  ${cost_per_call:.5f}')

n_images = len(df)
total_cost = cost_per_call * n_images
print(f'Total for {n_images} images:  ${total_cost:.3f}')
print(f'\n--- With prompt caching (subtopics cached) ---')
# Cached tokens billed at $0.30/M instead of $3/M
cached_tokens = len(subtopic_text)//4 + FIXED_OVERHEAD_TOKENS
uncached_tokens = int(char_stats.mean()/4)
cost_cached = (cached_tokens / 1_000_000 * 0.30) + (uncached_tokens / 1_000_000 * 3.00) + (mean_output / 1_000_000 * 15.00)
print(f'Cost per image with caching: ${cost_cached:.5f}')
print(f'Total for {n_images} images with caching: ${cost_cached * n_images:.3f}')
