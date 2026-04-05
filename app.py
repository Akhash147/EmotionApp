# pylint: disable=all
import re
import pickle
import traceback
import numpy as np
import pandas as pd
import nltk
import scipy.sparse as sp
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt_tab', quiet=True)

app = Flask(__name__)
CORS(app)


# Load Models
print("Loading models...")
rf_model = pickle.load(open('models/rf_model.pkl', 'rb'))
tfidf = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
le = pickle.load(open('models/label_encoder.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
intervention_db = pd.read_csv('models/intervention_knowledge_base.csv')

N_FEATURES = rf_model.n_features_in_

# Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
analyzer = SentimentIntensityAnalyzer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 2
    ]
    return ' '.join(tokens)

def prepare_features(text, cleaned_text):
    # Step 1 — TF-IDF on cleaned text
    tfidf_vec = tfidf.transform([cleaned_text])

    # Step 2 — Extract 12 numerical features
    vader_scores = analyzer.polarity_scores(text)
    blob = TextBlob(text)

    numerical = np.array([[
        float(len(text)),                          # text_length
        float(len(text.split())),                  # word_count
        float(text.count('!')),                    # exclamation_count
        float(text.count('?')),                    # question_count
        float(sum(1 for c in text if c.isupper()) / max(len(text), 1)),  # caps_ratio
        float(vader_scores['pos']),                # vader_positive
        float(vader_scores['neg']),                # vader_negative
        float(vader_scores['neu']),                # vader_neutral
        float(vader_scores['compound']),           # vader_compound
        float(blob.sentiment.subjectivity),        # subjectivity
        float(1),                                  # emotion_intensity placeholder
        float(100)                                 # subreddit_activity placeholder
    ]], dtype=np.float32)

    # Step 3 — Scale numerical features
    numerical_scaled = scaler.transform(numerical)

    # Step 4 — Combine TF-IDF + numerical
    import scipy.sparse as sp
    numerical_sparse = sp.csr_matrix(numerical_scaled)
    combined = sp.hstack([tfidf_vec, numerical_sparse])

    print(f"Combined shape: {combined.shape}")
    print(f"RF expects: {N_FEATURES}")

    # Step 5 — Match exact feature count
    combined_cols = combined.shape[1]
    if combined_cols >= N_FEATURES:
        final = combined[:, :N_FEATURES].toarray()
    else:
        padding = sp.csr_matrix((1, N_FEATURES - combined_cols))
        final = sp.hstack([combined, padding]).toarray()

    return final.astype(np.float32)

# Add right after loading models
print(f"TF-IDF vocabulary size: {len(tfidf.vocabulary_)}")
print(f"RF model expects: {rf_model.n_features_in_} features")
print(f"Label classes: {list(le.classes_)}")

# Test with a sample — pass both text and cleaned
test_text = "I am very angry about this situation"
test_clean = clean_text(test_text)
test_features = prepare_features(test_text, test_clean)
test_pred = rf_model.predict(test_features)[0]
test_emotion = le.inverse_transform([test_pred])[0]
print(f"Test prediction: {test_emotion}")

print(f" All models loaded — RF expects {N_FEATURES} features")
print(f" Emotion classes: {list(le.classes_)}")


# Community Mapping — Unique Per Emotion
community_map = {
    'anger': {
        'communities': [
            {'name': 'Political Communities',
             'reason': 'Politically charged content triggers intense backlash and heated debates',
             'severity': 92, 'icon': 'P'},
            {'name': 'News & Media Platforms',
             'reason': 'Controversial content spreads rapidly through news sharing networks',
             'severity': 85, 'icon': 'N'},
            {'name': 'Social Justice Groups',
             'reason': 'Injustice-related content provokes strong advocacy and protest responses',
             'severity': 78, 'icon': 'S'}
        ],
        'spread': [
            {'time': 'Within 30 minutes', 'community': 'Twitter & Reddit Political Forums', 'probability': 92},
            {'time': 'Within 2 hours', 'community': 'News Media & Journalists', 'probability': 78},
            {'time': 'Within 5 hours', 'community': 'General Public Social Media', 'probability': 65}
        ],
        'precautions': [
            'Issue a calm and factual public statement within 1 hour',
            'Avoid defensive or inflammatory responses online',
            'Engage community moderators to reduce escalation',
            'Monitor trending hashtags for further spread signals',
            'Prepare a dedicated crisis communication team immediately'
        ]
    },
    'annoyance': {
        'communities': [
            {'name': 'General Discussion Forums',
             'reason': 'Repetitive or irritating content creates community friction and disputes',
             'severity': 60, 'icon': 'G'},
            {'name': 'Tech & Developer Communities',
             'reason': 'Misinformation about technology deeply frustrates expert communities',
             'severity': 55, 'icon': 'T'},
            {'name': 'Hobby & Interest Groups',
             'reason': 'Off-topic or irrelevant content disrupts tightly focused communities',
             'severity': 40, 'icon': 'H'}
        ],
        'spread': [
            {'time': 'Within 1 hour', 'community': 'Reddit Subreddits & Forums', 'probability': 65},
            {'time': 'Within 3 hours', 'community': 'Tech Twitter & LinkedIn', 'probability': 50},
            {'time': 'Within 8 hours', 'community': 'General Discussion Boards', 'probability': 38}
        ],
        'precautions': [
            'Acknowledge the community concern with empathy',
            'Avoid repeating or amplifying the irritating content',
            'Clarify any misunderstandings with a factual response',
            'Allow community moderators to address the friction',
            'Monitor the situation for 24 hours before escalating'
        ]
    },
    'fear': {
        'communities': [
            {'name': 'Health & Wellness Communities',
             'reason': 'Health-related fears spread rapidly and trigger widespread panic responses',
             'severity': 90, 'icon': 'H'},
            {'name': 'Parents & Family Networks',
             'reason': 'Safety and wellbeing threats trigger immediate protective community responses',
             'severity': 86, 'icon': 'F'},
            {'name': 'Local Neighborhood Groups',
             'reason': 'Local threats cause immediate anxiety across geographically connected communities',
             'severity': 80, 'icon': 'L'}
        ],
        'spread': [
            {'time': 'Within 20 minutes', 'community': 'WhatsApp & Family Group Chats', 'probability': 90},
            {'time': 'Within 1 hour', 'community': 'Health Forums & Medical Communities', 'probability': 82},
            {'time': 'Within 4 hours', 'community': 'Local Facebook & Nextdoor Groups', 'probability': 70}
        ],
        'precautions': [
            'Publish verified and authoritative factual information immediately',
            'Partner with trusted health or government authorities to counter panic',
            'Set up a dedicated FAQ page addressing the specific fears raised',
            'Avoid sensational or alarming language in all communications',
            'Monitor mental health impact in affected communities and provide support resources'
        ]
    },
    'sadness': {
        'communities': [
            {'name': 'Mental Health Communities',
             'reason': 'Emotionally heavy content significantly impacts vulnerable and sensitive members',
             'severity': 84, 'icon': 'M'},
            {'name': 'Grief & Loss Support Groups',
             'reason': 'Tragic content resonates deeply within bereavement and support networks',
             'severity': 79, 'icon': 'G'},
            {'name': 'Youth & Student Communities',
             'reason': 'Young audiences are particularly emotionally sensitive to distressing news',
             'severity': 72, 'icon': 'Y'}
        ],
        'spread': [
            {'time': 'Within 1 hour', 'community': 'Mental Health Subreddits & Forums', 'probability': 80},
            {'time': 'Within 3 hours', 'community': 'Twitter Emotional Communities', 'probability': 68},
            {'time': 'Within 6 hours', 'community': 'Student & Youth Social Platforms', 'probability': 55}
        ],
        'precautions': [
            'Launch a compassionate mental health awareness campaign immediately',
            'Share supportive and empathetic community messaging',
            'Connect affected communities with professional support resources',
            'Avoid trivializing or dismissing the emotional concerns raised',
            'Engage qualified mental health professionals for guidance and response'
        ]
    },
    'joy': {
        'communities': [
            {'name': 'Entertainment & Fan Communities',
             'reason': 'Positive and exciting content amplifies enthusiasm and shared celebration',
             'severity': 28, 'icon': 'E'},
            {'name': 'Sports Fan Networks',
             'reason': 'Victory and celebration content spreads rapidly and enthusiastically among fans',
             'severity': 22, 'icon': 'S'},
            {'name': 'Creative & Artist Communities',
             'reason': 'Inspiring and uplifting content motivates creative expression and sharing',
             'severity': 18, 'icon': 'C'}
        ],
        'spread': [
            {'time': 'Within 30 minutes', 'community': 'Instagram & TikTok Fan Pages', 'probability': 75},
            {'time': 'Within 2 hours', 'community': 'Twitter Trending Topics', 'probability': 60},
            {'time': 'Within 5 hours', 'community': 'YouTube & Content Creator Communities', 'probability': 45}
        ],
        'precautions': [
            'Amplify the positive message authentically across all channels',
            'Engage actively with the community to sustain positive momentum',
            'Share genuine stories and testimonials related to the positive event',
            'Use this positive wave to strengthen brand trust and community bonds',
            'Create meaningful follow-up content to maintain engagement and goodwill'
        ]
    },
    'disgust': {
        'communities': [
            {'name': 'Ethics & Morality Communities',
             'reason': 'Content violating moral standards triggers immediate collective outrage',
             'severity': 87, 'icon': 'E'},
            {'name': 'Environmental Activist Groups',
             'reason': 'Environmental harm content causes intense reactions in conservation communities',
             'severity': 82, 'icon': 'V'},
            {'name': 'Consumer Rights Organizations',
             'reason': 'Unethical business or product behavior triggers widespread consumer disgust',
             'severity': 76, 'icon': 'C'}
        ],
        'spread': [
            {'time': 'Within 45 minutes', 'community': 'Twitter Activist Networks', 'probability': 88},
            {'time': 'Within 2 hours', 'community': 'Reddit Ethics & Morality Forums', 'probability': 75},
            {'time': 'Within 5 hours', 'community': 'Facebook Community & Advocacy Groups', 'probability': 62}
        ],
        'precautions': [
            'Acknowledge the community concern publicly and with complete sincerity',
            'Take immediate and clearly visible corrective action',
            'Avoid minimizing, deflecting, or dismissing the ethical concerns raised',
            'Publish a detailed and transparent accountability statement',
            'Engage independent ethics experts to review and provide credible response'
        ]
    },
    'surprise': {
        'communities': [
            {'name': 'Breaking News Communities',
             'reason': 'Unexpected and shocking events drive extremely rapid information sharing',
             'severity': 65, 'icon': 'B'},
            {'name': 'Technology Early Adopter Networks',
             'reason': 'Surprising technological innovations spark intense and widespread discussion',
             'severity': 58, 'icon': 'T'},
            {'name': 'General Social Media Users',
             'reason': 'Shocking and unexpected content achieves viral spread across all platforms',
             'severity': 62, 'icon': 'G'}
        ],
        'spread': [
            {'time': 'Within 15 minutes', 'community': 'Twitter Trending & Breaking News', 'probability': 82},
            {'time': 'Within 1 hour', 'community': 'Tech & Innovation Communities', 'probability': 65},
            {'time': 'Within 3 hours', 'community': 'General Public & Mainstream Media', 'probability': 52}
        ],
        'precautions': [
            'Provide clear and detailed context around the surprising information immediately',
            'Anticipate follow-up questions and prepare comprehensive answers in advance',
            'Monitor initial community reactions before crafting a broad response',
            'Proactively clarify any misunderstandings before they escalate further',
            'Use only official and verified channels for all external communications'
        ]
    },
    'excitement': {
        'communities': [
            {'name': 'Fan & Enthusiast Communities',
             'reason': 'Exciting announcements trigger immediate and enthusiastic community engagement',
             'severity': 35, 'icon': 'F'},
            {'name': 'Product & Brand Followers',
             'reason': 'Exciting product or brand news spreads rapidly among loyal followers',
             'severity': 30, 'icon': 'P'},
            {'name': 'Gaming & Entertainment Networks',
             'reason': 'Exciting gaming or entertainment content drives immediate viral sharing',
             'severity': 28, 'icon': 'G'}
        ],
        'spread': [
            {'time': 'Within 20 minutes', 'community': 'Discord & Gaming Communities', 'probability': 80},
            {'time': 'Within 1 hour', 'community': 'Twitter & Instagram Fan Pages', 'probability': 68},
            {'time': 'Within 3 hours', 'community': 'YouTube & Streaming Platforms', 'probability': 52}
        ],
        'precautions': [
            'Capitalize on the excitement with timely and engaging content',
            'Actively interact with enthusiastic community members',
            'Share exclusive or behind-the-scenes content to deepen excitement',
            'Use the positive energy to build long-term community relationships',
            'Avoid over-promising — ensure the excitement is backed by real substance'
        ]
    },
    'gratitude': {
        'communities': [
            {'name': 'Brand Loyalty Communities',
             'reason': 'Grateful content strengthens existing brand and community relationships',
             'severity': 20, 'icon': 'B'},
            {'name': 'Charitable & Volunteer Networks',
             'reason': 'Gratitude and appreciation content resonates deeply in giving communities',
             'severity': 18, 'icon': 'C'},
            {'name': 'Professional Networks',
             'reason': 'Appreciation content drives positive engagement in professional communities',
             'severity': 15, 'icon': 'P'}
        ],
        'spread': [
            {'time': 'Within 1 hour', 'community': 'LinkedIn & Professional Networks', 'probability': 60},
            {'time': 'Within 3 hours', 'community': 'Facebook Community Groups', 'probability': 48},
            {'time': 'Within 8 hours', 'community': 'General Social Media Platforms', 'probability': 35}
        ],
        'precautions': [
            'Acknowledge and genuinely respond to the gratitude expressed',
            'Share the positive feedback authentically with your broader community',
            'Use this moment to reinforce your core community values publicly',
            'Encourage other community members to share similar positive experiences',
            'Continue delivering the quality that generated the gratitude'
        ]
    },
    'neutral': {
        'communities': [
            {'name': 'General Discussion Communities',
             'reason': 'Neutral content generates minimal emotional reaction across broad communities',
             'severity': 15, 'icon': 'G'},
            {'name': 'Information & Research Networks',
             'reason': 'Factual and neutral content is consumed analytically without strong reactions',
             'severity': 12, 'icon': 'I'},
            {'name': 'Academic & Professional Groups',
             'reason': 'Objective information is calmly evaluated and discussed in analytical communities',
             'severity': 10, 'icon': 'A'}
        ],
        'spread': [
            {'time': 'Within 6 hours', 'community': 'Academic & Research Forums', 'probability': 30},
            {'time': 'Within 12 hours', 'community': 'Professional LinkedIn Networks', 'probability': 22},
            {'time': 'Within 24 hours', 'community': 'General News Aggregators', 'probability': 15}
        ],
        'precautions': [
            'Continue standard community monitoring procedures',
            'No immediate intervention required at this time',
            'Log this content for future trend analysis purposes',
            'Review again if community engagement increases significantly',
            'Maintain your normal communication and content cadence'
        ]
    }
}

def get_emotion_data(emotion):
    if emotion in community_map:
        return community_map[emotion]
    # Find closest match
    for key in community_map:
        if key in emotion or emotion in key:
            return community_map[key]
    return community_map['neutral']

# Predict Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        cleaned = clean_text(text)
        print(f"Cleaned text: '{cleaned}'")

        if not cleaned:
            return jsonify({'error': 'Text too short after cleaning'}), 400

        # Prepare features — pass both original and cleaned
        features = prepare_features(text, cleaned)
        print(f"Final feature shape: {features.shape}")

        # Predict
        prediction = rf_model.predict(features)[0]
        probabilities = rf_model.predict_proba(features)[0]
        emotion = le.inverse_transform([prediction])[0]
        print(f"Predicted Emotion: {emotion}")

        # Top 3 emotions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_emotions = [
            {
                'emotion': le.classes_[i],
                'probability': round(float(probabilities[i]) * 100, 1)
            }
            for i in top_indices
        ]

        # VADER
        vader = analyzer.polarity_scores(text)

        # Get emotion specific data
        emotion_data = get_emotion_data(emotion)

        # Intervention
        row = intervention_db[intervention_db['emotion'] == emotion]
        if len(row) > 0:
            intervention = row.iloc[0]['intervention']
            urgency = row.iloc[0]['urgency']
            response_time = row.iloc[0]['response_time']
        else:
            intervention = "Monitor and observe community activity"
            urgency = "LOW"
            response_time = "Within 48 hours"

        return jsonify({
            'emotion': emotion,
            'top_emotions': top_emotions,
            'urgency': urgency,
            'intervention': intervention,
            'response_time': response_time,
            'communities': emotion_data['communities'],
            'spread_timeline': emotion_data['spread'],
            'precautions': emotion_data['precautions'],
            'vader_compound': round(vader['compound'], 3),
            'sentiment': 'Positive' if vader['compound'] > 0.05
                        else 'Negative' if vader['compound'] < -0.05
                        else 'Neutral'
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/debug', methods=['GET'])
def debug():
    test_posts = [
        "This government is completely destroying our country!",
        "I just got promoted! Best day of my life!",
        "Scientists warn this virus could kill millions",
        "The new iPhone looks amazing I want one",
        "I hate how nobody listens to anyone anymore"
    ]

    results = []
    for post in test_posts:
        try:
            cleaned = clean_text(post)
            features = prepare_features(post, cleaned)
            prediction = rf_model.predict(features)[0]
            probabilities = rf_model.predict_proba(features)[0]
            emotion = le.inverse_transform([prediction])[0]

            top_indices = np.argsort(probabilities)[-3:][::-1]
            top_emotions = [
                {
                    'emotion': le.classes_[i],
                    'probability': round(
                        float(probabilities[i]) * 100, 1)
                }
                for i in top_indices
            ]

            results.append({
                'post': post,
                'emotion': emotion,
                'top_emotions': top_emotions
            })
        except Exception as e:
            results.append({
                'post': post,
                'error': str(e)
            })

    return jsonify(results)
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)