# import nltk
# nltk.download('stopwords')

# nltk.download('en_core_web_sm')
# nltk.download('words')

from pyresparser import ResumeParser

data = ResumeParser('/home/samyak/table_data_extract/HETVI_JULASANA.pdf').get_extracted_data()

