from lxml import etree

# Item
class Item:
    def __init__(self, seqNum, shotId):
        self.seqNum = seqNum
        self.shotId = shotId

    def to_xml_element(self):
        _element = etree.Element('item')
        _element.attrib["seqNum"] = self.seqNum
        _element.attrib["shotId"] = self.shotId
        return _element


# ########################################################################################## #
# Feature Result
class VideoFeatureExtractionFeatureResult:
    def __init__(self, fNum="-1", elapsedTime="-1", items=[], itemNames=[]):
        self.fNum = fNum
        self.elapsedTime = elapsedTime
        self.items = items

        if self.items == []:
            for _i in range(len(itemNames)):
                _item = Item(str(_i+1), itemNames[_i])
                self.items.append(_item)

    def to_xml_element(self):
        _element = etree.Element('videoFeatureExtractionFeatureResult')
        _element.attrib["fNum"] = self.fNum
        _element.attrib["elapsedTime"] = self.elapsedTime

        for _item in self.items:
            _element.append(_item.to_xml_element())
        return _element


# ########################################################################################## #
# Run Result
class VideoFeatureExtractionRunResult:
    def __init__(self, trType, pid, priority, desc, videoFeatureExtractionFeatureResults=[]):
        self.trType = trType
        self.pid = pid
        self.priority = priority
        self.desc = desc
        self.videoFeatureExtractionFeatureResults = videoFeatureExtractionFeatureResults

    def to_xml_element(self):
        _element = etree.Element("videoFeatureExtractionRunResult")
        _element.attrib["trType"] = self.trType
        _element.attrib["pid"] = self.pid
        _element.attrib["priority"] = self.priority
        _element.attrib["desc"] = self.desc
        for _videoFeatureExtractionFeatureResult in self.videoFeatureExtractionFeatureResults:
            _element.append(_videoFeatureExtractionFeatureResult.to_xml_element())
        return _element


# ########################################################################################## #
# Results
class VideoFeatureExtractionResults:
    def __init__(self, videoFeatureExtractionRunResults=[]):
        self.videoFeatureExtractionRunResults = videoFeatureExtractionRunResults

    def to_xml_element(self):
        _element = etree.Element("videoFeatureExtractionResults")
        for _videoFeatureExtractionRunResult in self.videoFeatureExtractionRunResults:
            _element.append(_videoFeatureExtractionRunResult.to_xml_element())
        return _element
