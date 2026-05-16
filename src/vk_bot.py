"""
Медицинский чат-бот ВКонтакте
Со сценарием диалога
"""
import vk_api
from vk_api.longpoll import VkLongPoll, VkEventType
from vk_api.keyboard import VkKeyboard, VkKeyboardColor
import random

TOKEN = "vk1.a._kH8stO725sQmEORerZI2YXFSKF4lEQ_jsJiUDkzU5lwY_4bUHYR7joZ4vJhSuS_NYK_Ta1TCBIr_mXy881wqLr8zr4JS9RjNeLnJbAXT0_xMMJIsdZT5hqu_pGHW4Qwvi3Ci5H-2-SynIJvEbpwkUS-IAxtIUjBif6exTD7RTJgjSzFtl4QQbf9LEs5C4jSfa_wJi_zSGCWWi7EPXgS_Q"
GROUP_ID = 238719922

vk_session = vk_api.VkApi(token=TOKEN)
vk = vk_session.get_api()
longpoll = VkLongPoll(vk_session)

print("✅ Бот ВКонтакте запущен! Напишите ему в группу.")

user_state = {}
user_answers = {}

STEP_SYMPTOMS = 0
STEP_TEMPERATURE = 1
STEP_DURATION = 2
STEP_COUGH = 3
STEP_RESULT = 4


def get_temp_keyboard():
    keyboard = VkKeyboard(one_time=True)
    keyboard.add_button("Да, температура есть", VkKeyboardColor.NEGATIVE)
    keyboard.add_button("Нет температуры", VkKeyboardColor.POSITIVE)
    keyboard.add_button("Не знаю", VkKeyboardColor.PRIMARY)
    return keyboard.get_keyboard()


def get_duration_keyboard():
    keyboard = VkKeyboard(one_time=True)
    keyboard.add_button("Сегодня", VkKeyboardColor.PRIMARY)
    keyboard.add_button("Несколько дней", VkKeyboardColor.PRIMARY)
    keyboard.add_line()
    keyboard.add_button("Неделю", VkKeyboardColor.PRIMARY)
    keyboard.add_button("Больше недели", VkKeyboardColor.PRIMARY)
    return keyboard.get_keyboard()


def get_cough_keyboard():
    keyboard = VkKeyboard(one_time=True)
    keyboard.add_button("Да, есть кашель", VkKeyboardColor.NEGATIVE)
    keyboard.add_button("Нет кашля", VkKeyboardColor.POSITIVE)
    return keyboard.get_keyboard()


def get_main_keyboard():
    keyboard = VkKeyboard(one_time=True)
    keyboard.add_button("Начать диагностику", VkKeyboardColor.PRIMARY)
    keyboard.add_button("Помощь", VkKeyboardColor.SECONDARY)
    return keyboard.get_keyboard()


def send_message(user_id, text, keyboard=None):
    vk.messages.send(
        user_id=user_id,
        message=text,
        random_id=random.randint(1, 2**31),
        keyboard=keyboard
    )

def make_diagnosis(answers):
    """
    Получает ответы пользователя и возвращает предполагаемый диагноз.
    ПОКА ЧТО — простая логика на if-else.
    ПОТОМ — замените на вызов модели.
    """
    symptoms = answers.get("symptoms", "")
    temp = answers.get("temperature", "")
    duration = answers.get("duration", "")
    cough = answers.get("cough", "")

    has_temp = "да" in temp.lower() or "есть" in temp.lower()
    has_cough = "да" in cough.lower() or "есть" in cough.lower()
    is_recent = "сегодня" in duration.lower() or "несколько" in duration.lower()

    result = "📊 Результат диагностики:\n\n"

    if has_temp and has_cough and is_recent:
        result += "🔴 Возможные причины:\n"
        result += "— ОРВИ (острая респираторная вирусная инфекция)\n"
        result += "— Грипп\n\n"
        result += "💊 Рекомендации:\n"
        result += "— Постельный режим\n"
        result += "— Обильное питьё\n"
        result += "— Жаропонижающее при t > 38°C\n"
        result += "— Обратиться к врачу при ухудшении\n"

    elif has_temp and not has_cough:
        result += "🟡 Возможные причины:\n"
        result += "— Вирусная инфекция без респираторных проявлений\n"
        result += "— Воспалительный процесс\n\n"
        result += "💊 Рекомендации:\n"
        result += "— Контроль температуры\n"
        result += "— Обильное питьё\n"
        result += "— Обратиться к терапевту\n"

    elif has_cough and not has_temp:
        result += "🟢 Возможные причины:\n"
        result += "— Аллергическая реакция\n"
        result += "— Бронхит (начальная стадия)\n"
        result += "— Раздражение дыхательных путей\n\n"
        result += "💊 Рекомендации:\n"
        result += "— Тёплое питьё\n"
        result += "— Избегать раздражителей\n"
        result += "— Обратиться к врачу если кашель более 3 дней\n"

    else:
        result += "🟢 Серьёзных симптомов не выявлено.\n"
        result += "Рекомендуется отдых и наблюдение.\n"
        result += "При ухудшении — обратиться к врачу.\n"

    result += "\n⚠️ Это предположение, а не окончательный диагноз. Обратитесь к врачу."

    return result


for event in longpoll.listen():
    if event.type == VkEventType.MESSAGE_NEW and event.to_me:
        user_id = event.user_id
        text = event.text.strip() if event.text else ""
        step = user_state.get(user_id, STEP_SYMPTOMS)
        answers = user_answers.get(user_id, {})


        if text.lower() in ["/start", "начать", "начать диагностику"]:
            user_state[user_id] = STEP_SYMPTOMS
            user_answers[user_id] = {}
            send_message(user_id,
                "🩺 Давайте проведём диагностику.\n\n"
                "Шаг 1 из 4: Опишите ваши симптомы.\n"
                "Например: «болит голова, слабость, насморк»",
                keyboard=get_main_keyboard()
            )

        elif text.lower() in ["/help", "помощь"]:
            send_message(user_id,
                "📋 Я задам 4 вопроса о ваших симптомах и выдам предположительный диагноз.\n\n"
                "⚠️ Это не заменяет визит к врачу!\n\n"
                "Команды:\n"
                "«Начать диагностику» — начать заново\n"
                "«Помощь» — эта подсказка\n"
                "«Сброс» — сбросить диалог",
                keyboard=get_main_keyboard()
            )

        elif text.lower() in ["/reset", "сброс"]:
            user_state[user_id] = STEP_SYMPTOMS
            user_answers[user_id] = {}
            send_message(user_id,
                "🔄 Диалог сброшен. Напишите «Начать диагностику» чтобы начать заново.",
                keyboard=get_main_keyboard()
            )


        elif step == STEP_SYMPTOMS:
            answers["symptoms"] = text
            user_answers[user_id] = answers
            user_state[user_id] = STEP_TEMPERATURE

            send_message(user_id,
                f"✅ Записал: «{text}»\n\n"
                "Шаг 2 из 4: Есть ли у вас температура?",
                keyboard=get_temp_keyboard()
            )

        elif step == STEP_TEMPERATURE:
            answers["temperature"] = text
            user_answers[user_id] = answers
            user_state[user_id] = STEP_DURATION

            send_message(user_id,
                f"✅ Ответ: «{text}»\n\n"
                "Шаг 3 из 4: Как давно появились симптомы?",
                keyboard=get_duration_keyboard()
            )

        elif step == STEP_DURATION:
            answers["duration"] = text
            user_answers[user_id] = answers
            user_state[user_id] = STEP_COUGH

            send_message(user_id,
                f"✅ Ответ: «{text}»\n\n"
                "Шаг 4 из 4: Есть ли у вас кашель?",
                keyboard=get_cough_keyboard()
            )

        elif step == STEP_COUGH:
            answers["cough"] = text
            user_answers[user_id] = answers
            user_state[user_id] = STEP_SYMPTOMS
            result = make_diagnosis(answers)

            send_message(user_id,
                f"✅ Ответ: «{text}»\n\n{result}\n\n"
                "Напишите «Начать диагностику» для нового опроса.",
                keyboard=get_main_keyboard()
            )

        else:
            send_message(user_id,
                "Напишите «Начать диагностику» чтобы начать.",
                keyboard=get_main_keyboard()
            )