const columnDefs = [
  { headerName:"Atlas Scaling Factor",field: "ASF"},
  { headerName:"Age",field: "Age" },
  { headerName:"Clinical Dementia Rating",field: "CDR"},
  { headerName:"Years of Education",field: "EDUC" },
  { headerName:"Demented or not",field: "Group" },
  { headerName:"Mini Mental State Examination",field: "MMSE" },
  { headerName:"Socioeconomic Status",field: "SES"},
  { headerName:"Estimated total intracranial volume",field: "eTIV"},
  { headerName:"0.0 as female, 1.0 as male",field: "gender"},
  { headerName:"Normalize Whole Brain Volume",field: "nWBV"}
  ];
  

  
  // let the grid know which columns and what data to use
  const gridOptions = {
    columnDefs: columnDefs,
    rowData: rowData
  };
  function onBtExport() {
    gridOptions.api.exportDataAsExcel();
  }
  // setup the grid after the page has finished loading
  document.addEventListener('DOMContentLoaded', () => {
      const gridDiv = document.querySelector('#myGrid');
      new agGrid.Grid(gridDiv, gridOptions);
  });