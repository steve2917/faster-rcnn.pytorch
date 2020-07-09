import methods.xmlExport.entities as entities
from lxml import etree

# Dummy Data
pid = "VCL"
trType = "O"

RESULT_LIST = [
  "Shot30_006",
  "Shot37_009",
  "Shot8_015",
  "Shot38_081",
  "Shot23_032",
  "Shot38_084",
  "Shot22_046",
  "Shot17_004",
  "Shot39_062",
  "Shot21_015",
  "Shot16_041",
  "Shot34_029",
  "Shot28_062",
  "Shot11_012",
  "Shot38_072",
  "Shot31_010",
  "Shot26_001",
  "Shot18_028",
  "Shot31_032",
  "Shot36_076",
  "Shot31_007"]

fNum = "_2"
elapsedTime = "_12"
priority = "1"
desc = "this is my description"


# Test with dummy data
if __name__ == '__main__':
    FeatureResults = []

    for fNum in range(1, 5):
        FeatureResult = entities.VideoFeatureExtractionFeatureResult(str(fNum), elapsedTime, itemNames=RESULT_LIST)
        FeatureResults.append(FeatureResult)

    RunResult = entities.VideoFeatureExtractionRunResult(trType, pid, priority, desc, FeatureResults)
    RunResults = [RunResult]
    Results = entities.VideoFeatureExtractionResults(RunResults)
    etree.ElementTree(Results.to_xml_element()).write(
        'export.xml',
        pretty_print=True,
        xml_declaration=True,
        encoding="ISO-8859-1",
        doctype='<!DOCTYPE videoFeatureExtractionResults SYSTEM "https://www-nlpir.nist.gov/projects/tv2020/dtds/videoFeatureExtractionResults.dtd">'
    )



