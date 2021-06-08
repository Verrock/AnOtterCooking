import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.response import open_in_browser


class MarmitonScraper(scrapy.Spider):
    name = 'MarmitonScraper'

    def __init__(self, category='', **kwargs):
        super(MarmitonScraper, self).__init__(**kwargs)
        self.start_urls = [f'https://www.marmiton.org/recettes/index/categorie/{category}']

    def parse(self, response, **kwargs):
        RECIPE_BLOCK_SELECTOR = '.recipe-card-link'
        RATING_SELECTOR = '.recipe-card__rating__value::text'
        NBR_AVIS = '.mrtn-font-discret::text'
        RECIPE_URL_SELECTOR = 'a::attr(href)'
        TITLE_SELECTOR = 'h4 ::text'
        NEXT_PAGE = 'li.selected~* a ::attr(href)'
        next_crawl = None
        item = {}

        for recipe in response.css(RECIPE_BLOCK_SELECTOR):

            nb_reviews = int(recipe.css(NBR_AVIS).extract_first().split(' ')[1])
            rating = float(recipe.css(RATING_SELECTOR).extract_first())
            next_crawl =  response.css(NEXT_PAGE).extract_first()

            #if rating > 3.5 and nb_reviews > 100:
            item['nb_reviews'] = nb_reviews
            item['rating'] = rating
            item['recipe'] = recipe.css(RECIPE_URL_SELECTOR).extract_first()
            with open('data_no_cond.csv', 'a') as f:
                f.write('nb_reviews:{0}, rating:{1}, recipe_url:{2} \n'.format(item['nb_reviews'], item['rating'], item['recipe']))
            self.parse_recipe(item['recipe'])
        if next_crawl:
            yield scrapy.Request(response.urljoin(next_crawl), callback=self.parse)


if __name__ == "__main__":
    process = CrawlerProcess(settings={
        "FEEDS": {
            "items.json": {"format": "json"}
        },
    })

    process.crawl(MarmitonScraper)
    process.start()