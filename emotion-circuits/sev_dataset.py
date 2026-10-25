# -*- coding: utf-8 -*-
"""
SEV-style dataset (Scenario-Event with Valence) for emotion-circuit extraction.

Original skeletons written for this repo, following the SEV idea from
"Do LLMs Feel? Emotion Circuits Discovery and Control" (arXiv:2510.11328):
a neutral scenario plus a positive / neutral / negative event, so the same
context can elicit different emotions and activations can be de-contextualized
(emotion activation minus the group mean of its skeleton).

Each skeleton produces, per emotion, an elicitation prompt asking the model
to continue as a first-person narrator feeling that emotion about the event
whose valence matches the emotion (positive -> happiness/surprise,
negative -> anger/sadness/fear/disgust). Neutral events are used for the
neutral baseline of every group.
"""

EMOTIONS = ["anger", "sadness", "happiness", "fear", "disgust", "surprise"]

# valence preferred by each emotion when picking the event of a skeleton
EMOTION_VALENCE = {
    "anger": "negative",
    "sadness": "negative",
    "fear": "negative",
    "disgust": "negative",
    "happiness": "positive",
    "surprise": "positive",  # surprise also works with negative; positive keeps groups balanced
}

# simple lexical families used to sanity-check the elicited text
EMOTION_LEXICON = {
    "anger": ["angry", "furious", "rage", "mad", "unfair", "fury", "irritated", "outraged"],
    "sadness": ["sad", "sorrow", "cry", "tears", "grief", "heartbroken", "mourn", "lonely", "down"],
    "happiness": ["happy", "joy", "delighted", "glad", "wonderful", "cheerful", "thrilled", "smile"],
    "fear": ["afraid", "scared", "fear", "terrified", "dread", "panic", "anxious", "frightened"],
    "disgust": ["disgusting", "gross", "revolting", "nasty", "repulsive", "sickening", "vile"],
    "surprise": ["surprised", "wow", "unbelievable", "astonished", "amazed", "unexpected", "shocked"],
}

SKELETONS = [
    {"theme": "Work/Job",
     "scenario": "I finished an important report and sent it to my manager yesterday",
     "event": {
         "positive": "My manager praised the report in front of the whole team and proposed me for a promotion.",
         "neutral": "My manager confirmed receipt of the report and scheduled a review meeting for next week.",
         "negative": "My manager dismissed the report without reading it and gave the credit to a colleague."}},
    {"theme": "Work/Job",
     "scenario": "I prepared for weeks to present my project at the company meeting",
     "event": {
         "positive": "The directors applauded the project and approved the budget on the spot.",
         "neutral": "The directors took notes during the presentation and said they would answer by email.",
         "negative": "The projector failed, nobody could see my slides, and the meeting was cancelled."}},
    {"theme": "Family",
     "scenario": "I organized a birthday dinner for my mother at her favorite restaurant",
     "event": {
         "positive": "She arrived, hugged me with tears of joy, and said it was her best birthday ever.",
         "neutral": "She thanked me for the dinner and we talked about ordinary things during the meal.",
         "negative": "She called an hour before to say she was not coming, without any explanation."}},
    {"theme": "Family",
     "scenario": "My brother promised to help me move to my new apartment on Saturday",
     "event": {
         "positive": "He arrived early with two friends and we finished everything before noon.",
         "neutral": "He arrived at the agreed time and we carried the boxes together as planned.",
         "negative": "He never showed up and did not answer my calls while I carried everything alone."}},
    {"theme": "Friends",
     "scenario": "I lent my best friend a large sum of money three months ago",
     "event": {
         "positive": "He paid me back in full today and invited me to dinner to thank me.",
         "neutral": "He told me he will pay me back at the end of the month as we agreed.",
         "negative": "I saw photos of him on an expensive trip while he keeps saying he cannot pay me."}},
    {"theme": "Friends",
     "scenario": "I told my closest friend a secret that was very important to me",
     "event": {
         "positive": "She kept the secret and supported me through the whole situation.",
         "neutral": "She listened carefully and changed the subject when others asked about it.",
         "negative": "The whole group now knows the secret because she told everyone at a party."}},
    {"theme": "Health",
     "scenario": "I went to the clinic to pick up the results of my medical tests",
     "event": {
         "positive": "The doctor smiled and told me everything is perfectly fine.",
         "neutral": "The doctor explained the results are within the normal ranges.",
         "negative": "The doctor asked me to sit down and said they found something that worries them."}},
    {"theme": "Health",
     "scenario": "I have been training for months to run the city marathon",
     "event": {
         "positive": "I crossed the finish line with my best time and the crowd cheered my name.",
         "neutral": "I finished the race in an average time, tired but satisfied.",
         "negative": "I injured my knee at kilometer ten and had to abandon the race."}},
    {"theme": "Money",
     "scenario": "I checked my bank account this morning before going to work",
     "event": {
         "positive": "I discovered an unexpected bonus payment from my company.",
         "neutral": "The balance shows the usual movements of the month.",
         "negative": "I discovered charges I never made and the account is almost empty."}},
    {"theme": "Money",
     "scenario": "I saved for two years to buy a used car",
     "event": {
         "positive": "The seller accepted my offer and the car passed the inspection perfectly.",
         "neutral": "The seller is considering my offer and will answer tomorrow.",
         "negative": "The car broke down on the way home and the seller refuses to answer."}},
    {"theme": "Daily life",
     "scenario": "I was cooking dinner for my guests who would arrive in an hour",
     "event": {
         "positive": "Everything turned out delicious and my guests asked for the recipes.",
         "neutral": "The dinner was ready on time and everyone ate without comments.",
         "negative": "I burned the main dish and the smoke alarm went off in front of everyone."}},
    {"theme": "Daily life",
     "scenario": "I took my dog for a walk in the park near my house",
     "event": {
         "positive": "He played happily with other dogs and learned a new trick.",
         "neutral": "We walked our usual route and came back home.",
         "negative": "He escaped, ran into the street, and I could not find him for hours."}},
    {"theme": "Studies",
     "scenario": "I studied all semester for the most difficult exam of my degree",
     "event": {
         "positive": "I got the highest grade of the class and the professor congratulated me.",
         "neutral": "I passed the exam with an average grade.",
         "negative": "I failed the exam by one point and must repeat the whole course."}},
    {"theme": "Studies",
     "scenario": "I submitted my thesis after a year of research",
     "event": {
         "positive": "The committee praised it as one of the best works of the year.",
         "neutral": "The committee accepted it with minor corrections.",
         "negative": "The committee rejected it and asked me to start over with a new topic."}},
    {"theme": "Relationships",
     "scenario": "I planned a surprise weekend trip with my partner",
     "event": {
         "positive": "My partner loved the surprise and said it was the best weekend of our relationship.",
         "neutral": "We spent a quiet weekend away from the city.",
         "negative": "My partner told me during the trip that they want to end the relationship."}},
    {"theme": "Relationships",
     "scenario": "I wrote a long letter to someone I had not spoken to in years",
     "event": {
         "positive": "They answered the same day, happy to rebuild the friendship.",
         "neutral": "They answered politely a week later.",
         "negative": "The letter came back unopened with no explanation."}},
    {"theme": "Home",
     "scenario": "I came home after a long day at work",
     "event": {
         "positive": "My neighbors had fixed my broken fence as a surprise gesture.",
         "neutral": "Everything was as I left it in the morning.",
         "negative": "A pipe had burst and the floor was covered in dirty water."}},
    {"theme": "Home",
     "scenario": "I planted tomatoes in my small garden in spring",
     "event": {
         "positive": "The plants grew strong and gave an abundant harvest.",
         "neutral": "The plants grew at a normal pace.",
         "negative": "A plague destroyed every plant just before the harvest."}},
]


def build_elicitation_messages(skeleton, emotion):
    """Chat messages that ask the model to continue as someone feeling `emotion`."""
    valence = EMOTION_VALENCE[emotion]
    event = skeleton["event"][valence]
    user = (
        f"Situation: {skeleton['scenario']}. Then this happened: {event}\n\n"
        f"Write a short first-person reaction (two sentences maximum) expressing intense {emotion}. "
        f"Do not explain, just react."
    )
    return [
        {"role": "system", "content": f"You are an actor. You feel intense {emotion} right now. Stay in character."},
        {"role": "user", "content": user},
    ]


def build_neutral_messages(skeleton):
    """Neutral baseline of the skeleton group: same scenario, neutral event, plain reaction."""
    user = (
        f"Situation: {skeleton['scenario']}. Then this happened: {skeleton['event']['neutral']}\n\n"
        f"Write a short first-person factual comment (two sentences maximum), without emotions."
    )
    return [
        {"role": "system", "content": "You are a precise, neutral observer."},
        {"role": "user", "content": user},
    ]


def text_passes_lexicon(text, emotion):
    """Cheap label check: the elicited text mentions the emotion's lexical family."""
    low = text.lower()
    return any(w in low for w in EMOTION_LEXICON[emotion])
