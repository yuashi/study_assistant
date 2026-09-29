from langchain_community.document_loaders import WebBaseLoader

url="https://www.apple.com/de/shop/buy-mac?afid=p240%7Cgo~cmp-234438559~adg-43678030514~ad-743513204116_kwd-987393509~dev-c~ext-~prd-~mca-~nt-search&cid=aos-de-kwgo-txt-mac-mac--"
data=WebBaseLoader(url)
docs=data.load()