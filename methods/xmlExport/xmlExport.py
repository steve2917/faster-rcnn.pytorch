import methods.xmlExport.entities as entities
from lxml import etree
import os


def export(export, elapsedTimeString, _work_dir='/home/demertzis/GitHub/tecData/tmp', priority='1', desc="multi classifier approach", pid="VCL", trType="L"):
    FeatureResults = []

    i = 0
    for fNum in export.keys():
        FeatureResult = entities.VideoFeatureExtractionFeatureResult(str(fNum), elapsedTimeString, items=[], itemNames=export[fNum])
        FeatureResults.append(FeatureResult)
        i += 1

    RunResult = entities.VideoFeatureExtractionRunResult(trType, pid, priority, desc, FeatureResults)
    RunResults = [RunResult]
    Results = entities.VideoFeatureExtractionResults(RunResults)
    etree.ElementTree(Results.to_xml_element()).write(
        os.path.join(_work_dir, 'xmlOutput.xml'),
        pretty_print=True,
        xml_declaration=True,
        encoding="ISO-8859-1",
        doctype='<!DOCTYPE videoFeatureExtractionResults SYSTEM "https://www-nlpir.nist.gov/projects/tv2020/dtds/videoFeatureExtractionResults.dtd">'
    )
