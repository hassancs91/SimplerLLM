import newspaper


#Load Files


def read_content_from_url(url):
    """
    Extracts the text content from a given URL using the newspaper package.

    Parameters:
    url (str): The URL of the article to extract text from.

    Returns:
    str: The text content of the article if extraction is successful, None otherwise.
    """
    try:
        article = newspaper.Article(url)
        article.download()

        if article.download_state == 2:
            article.parse()
            return article.text
        else:
            #logging.error(f"Unable to download article from URL: {url}")
            return None
    except newspaper.ArticleException as e:
        #logging.error(f"Error while processing URL {url}: {e}")
        return None
    