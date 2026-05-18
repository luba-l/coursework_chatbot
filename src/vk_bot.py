import re
import random
import joblib
import spacy
import vk_api
from vk_api.longpoll import VkLongPoll, VkEventType
from vk_api.keyboard import VkKeyboard, VkKeyboardColor
import os
from dotenv import load_dotenv
load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")

vk_session = vk_api.VkApi(token=TOKEN)
vk = vk_session.get_api()
longpoll = VkLongPoll(vk_session)

print("Бот запущен")

model = joblib.load("models/best_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

nlp = spacy.load("ru_core_news_sm")

waiting_for_symptoms = {}
recommendations = {
    "простуда": [
        "Покой и сон",
        "Тёплое питьё",
        "Проветривание помещения"
    ],
    "пневмония": [
        "Срочно обратиться к врачу",
        "Не переносить болезнь на ногах",
        "Контролировать температуру"
    ]
}

def get_main_keyboard():
    keyboard = VkKeyboard(one_time=True)
    keyboard.add_button(
        "Начать диагностику",
        VkKeyboardColor.PRIMARY
    )
    keyboard.add_button(
        "Помощь",
        VkKeyboardColor.SECONDARY
    )
    return keyboard.get_keyboard()

def send_message(user_id, text, keyboard=None):
    vk.messages.send(
        user_id=user_id,
        message=text,
        random_id=random.randint(1, 2**31),
        keyboard=keyboard
    )

def clean_text(text):
    """Очистка текста от шума"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^а-яё\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lemmatize_text(text):
    """Лемматизация текста с помощью spaCy"""
    if not isinstance(text, str) or not text:
        return ""
    try:
        doc = nlp(text)
        return ' '.join([token.lemma_ for token in doc if not token.is_punct])
    except:
        return text

def predict_top3(text):
    cleaned = clean_text(text)
    lemmatized = lemmatize_text(cleaned)
    X = vectorizer.transform([lemmatized])
    decision = model.decision_function(X)[0]
    classes = model.classes_
    pairs = list(zip(classes, decision))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:3]

for event in longpoll.listen():
    if event.type == VkEventType.MESSAGE_NEW and event.to_me:
        user_id = event.user_id
        text = event.text.lower()
        if text == "начать диагностику":
            waiting_for_symptoms[user_id] = True
            send_message(
                user_id,
                "Опишите ваши симптомы одним сообщением.",
                keyboard=get_main_keyboard()
            )

        elif text == "помощь":
            send_message(
                user_id,
                "Напишите симптомы одним сообщением.\n\n"
                "Пример:\n"
                "болит голова, температура, слабость",
                keyboard=get_main_keyboard()
            )

        elif waiting_for_symptoms.get(user_id):
            top3 = predict_top3(text)
            best_diagnosis = top3[0][0]
            result = "📊 Возможные диагнозы:\n\n"
            for i, (disease, score) in enumerate(top3, start=1):
                result += f"{i}. {disease}\n"
            result += f"\n💊 Рекомендации для заболевания {best_diagnosis}:\n"
            if best_diagnosis in recommendations:
                for rec in recommendations[best_diagnosis]:
                    result += f"• {rec}\n"
            else:
                result += (
                    "• Обратитесь к врачу для консультации\n"
                    "• Не занимайтесь самолечением\n"
                )
            result += (
                "\n\n⚠️ Бот не заменяет врача."
            )
            send_message(
                user_id,
                result,
                keyboard=get_main_keyboard()
            )
            waiting_for_symptoms[user_id] = False

        else:
            send_message(
                user_id,
                "Нажмите «Начать диагностику».",
                keyboard=get_main_keyboard()
            )