import os
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
except Exception:
    client = None

def summarize_insight(text):
    if not text:
        return 'No input text.'
    if client is None:
        # simple fallback summarizer
        s = text.strip()
        return '[no-openai] Quick summary: ' + (s[:200] + ('...' if len(s) > 200 else ''))
    try:
        resp = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{'role':'user','content':f'Summarize this into a short market insight: {text}'}],
            max_tokens=150
        )
        return resp.choices[0].message.content
    except Exception as e:
        return '[openai-failed] ' + str(e)
