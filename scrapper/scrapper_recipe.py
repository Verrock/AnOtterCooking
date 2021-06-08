import scrapy
import pandas as pd
import re
from scrapy.crawler import CrawlerProcess


class RecipeScraper(scrapy.Spider):
    name = 'RecipeScraper'
    count = 0
    url_list = pd.read_csv('data_no_cond.csv')
    recipe_url_to_split = list(url_list.iloc[:, [2]].values[:, 0])
    test_url = [url.split('recipe_url:')[1].strip() for url in recipe_url_to_split]

    def __init__(self, category='', **kwargs):
        super(RecipeScraper, self).__init__(**kwargs)
        #self.start_urls = [f'https://www.marmiton.org/recettes/recette_soupe-au-chou-vert_54188.aspx']  # self.test_url
        self.start_urls = self.test_url

    def parse(self, response, **kwargs):
        ING_BLOCK_SELECTOR = '.recipe-ingredients__list__item'
        ING_QT = '.recipe-ingredient-qt::text'
        ING = '.ingredient::text'
        ING_CPL = '.recipe-ingredient__complement::text'
        NB_TITLE = '.recipe-infos__item-title::text'
        NB = '.recipe-infos__quantity__value::text'
        # TODO : retour sur liste empty (retirer liste en compréhension) et séparer ing + scale avec split("de", 1)
        items = []

        s_url = response.url
        pattern = re.compile(".*\_([0-9]*)\.")
        id = pattern.match(s_url).group(1)

        for i in response.css(ING_BLOCK_SELECTOR):
            res_split = i.css(ING).extract_first().split("de", 1)
            item = {'Quantity': i.css(ING_QT).extract_first(),
                    'Ingredient_cpl': i.css(ING_CPL).extract_first().strip()}
            if res_split[0] is not None:
                item['Ingredient'] = res_split[0].strip()
            if len(res_split) >= 2:
                item['Scale'] = res_split[0].strip()
                item['Ingredient'] = res_split[1].strip()
            for i in response.css('.recipe-infos__quantity'):
                item['Unit'] =  i.css(NB_TITLE).extract_first()
                item['Number'] = i.css(NB).extract_first()
                item['ID'] = id
            items.append(item)
        df = pd.DataFrame(items)
        df.to_csv('final.csv', index=False, encoding='utf-8', header=False,
                  columns=['Quantity', 'Scale', 'Ingredient',
                           'Ingredient_cpl', 'Number', 'Unit', 'ID'], mode='a')
