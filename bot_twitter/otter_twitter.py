import json
import requests
import tweepy
import logging
import time
import pandas as pd
import random

API_TOKEN = "api_qCwxkiMuuALxfWQNedvcOoewtZGkhWLEDZ"
API_URL = "https://api-inference.huggingface.co/models/antoiloui/belgpt2"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
# API Key : j6u5CdPS45KsWWFOJhr9VyMI7
# API Secret Key : U2vFUwUGvvg0j0ktv4RQEjBwPGG3YiStduaSv3yIsPIavyCN90
# Bearer Token : AAAAAAAAAAAAAAAAAAAAAN%2FSNgEAAAAA8Vy8ADax%2Bk2vpOdSfd%2Fok0l%2FEgE%3DpABKOQoqKluc3BMqgSqeqSCOaMRQhOOdxgjXra0yf3PeJ7YYjM
# Access token : 1371851932202106890-PuRX1i4LLUU51SqPWdNgCetX3eHGeA
# Access token secret : e2iRPfOij3KUSLpK7buZDWgHzOGjbSLRTM4sJ2smtzt6c


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def call_model_recipe():
    API_TOKEN = "api_qCwxkiMuuALxfWQNedvcOoewtZGkhWLEDZ"
    API_URL = "https://api-inference.huggingface.co/models/antoiloui/belgpt2"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    def query(payload):
        data = json.dumps(payload)
        response = requests.request("POST", API_URL, headers=headers, data=data)
        return json.loads(response.content.decode("utf-8"))

    df = pd.read_csv('final.csv')
    df['Scale'].fillna('', inplace=True)
    df['Ingredient'].fillna('', inplace=True)
    df['Instruction'] = df['Quantity'] + ' ' + df['Scale'].astype(str) + ' ' + df['Ingredient'].astype(str)
    # df['Nombre'] = "pour " + df['Number'].astype(str) + ' ' + df['Unit'].astype(str)
    df = df.drop(['ID', 'Ingredient', 'Scale', 'Ingredient_cpl', 'Quantity', 'Number', 'Unit'], axis=1)
    df = df.dropna()
    verbs = ["Chauffez", "Réduire", "Beurrez", "Mettre au four", "Fondre", "Caramélisez", "Couper", "Pressez",
             "Pétrissez"]
    recette = []
    ingredients = []
    for i in range(7):
        ingredients.append(df.iloc[random.randint(0, 89373)][:].values)

    for i in range(6):
        ing = verbs[random.randint(0, 8)] + ' ' + ingredients[i][0]
        data = query(ing)
        t = data[0]['generated_text'].split('.')
        recette.append(t[0])
    final_recipe = '\n'.join(recette)
    if len(final_recipe) > 540:
        call_model_recipe()
    return final_recipe


def check_mentions(api, keywords, since_id):
    logger.info("Retrieving mentions")
    new_since_id = since_id
    for tweet in tweepy.Cursor(api.mentions_timeline, since_id=since_id).items():
        new_since_id = max(tweet.id, new_since_id)
        if tweet.in_reply_to_status_id is not None:
            continue
        if any(keyword in tweet.text.lower() for keyword in keywords):
            logger.info(f"Answering to {tweet.user.name}")

            if not tweet.user.following:
                tweet.user.follow()

            recette = call_model_recipe()
            print(int(len(recette)/2))
            if len(recette) > 270:
                recipe1, recipe2 = recette[:int(len(recette)/2)], recette[int(len(recette)/2):]
                api.update_status(
                    status="@" + tweet.user.screen_name + ' ' + recipe1,
                    in_reply_to_status_id=tweet.id,
                )

                api.update_status(
                    status="@" + tweet.user.screen_name + ' ' + recipe2,
                    in_reply_to_status_id=tweet.id,
                )


            else:
                api.update_status(
                    status= "@"+ tweet.user.screen_name + ' ' + recette,
                    in_reply_to_status_id=tweet.id,
                )
    return new_since_id

def follow_followers(api):
    logger.info("Retrieving and following followers")
    for follower in tweepy.Cursor(api.followers).items():
        if not follower.following:
            logger.info(f"Following {follower.name}")
            follower.follow()

def main():
    # Authenticate to Twitter
    auth = tweepy.OAuthHandler("j6u5CdPS45KsWWFOJhr9VyMI7", "U2vFUwUGvvg0j0ktv4RQEjBwPGG3YiStduaSv3yIsPIavyCN90")
    auth.set_access_token("1371851932202106890-PuRX1i4LLUU51SqPWdNgCetX3eHGeA",
                          "e2iRPfOij3KUSLpK7buZDWgHzOGjbSLRTM4sJ2smtzt6c")

    # Create API Object
    api = tweepy.API(auth, wait_on_rate_limit=True,
    wait_on_rate_limit_notify=True)

    since_id = 1
    while True:
        follow_followers(api)
        since_id = check_mentions(api, ["!recette"], since_id)
        logger.info("Waiting...")
        time.sleep(60)


if __name__ == "__main__":
    main()