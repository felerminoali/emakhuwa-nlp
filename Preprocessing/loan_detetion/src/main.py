from loanwords import LoanwordModel, LexicalBased, ClassifierBased


if __name__ == "__main__":

    # model = LoanwordModel(LexicalBased())   
    # result = model.detect("nenna vaa , yohaana edireito y ’ eproteção ni okhaviheriwa n ’ estado .")
    # print(result)

    model = LoanwordModel(ClassifierBased())   
    result = model.detect("maxakha oxipiritali ahomwiihana")
    print(result)


