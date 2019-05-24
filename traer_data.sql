declare      @clientId smallint,

       @policyId smallint,

       @from varchar(10),

       @to varchar(10)

       set @clientId = 133

       set @policyId = 1

       set    @from = N'20190324'

       set    @to = N'20190524'

       select executionId,convert(varchar(500),executionDate,103) executionDate,cast(cast(executionDate as time) as varchar(5)) executionDateTime,policyId,isnull(policyVersion,0) policyVersion into #tmp from rulepolicyexecutions

       where clientId = @clientid and (policyId = @policyid or @policyId = -1) and DATEADD(dd, DATEDIFF(dd, 0, executionDate), 0) between @from and @to

 

       create index ix_id on #tmp(executionid)

 

       declare ors cursor for(select distinct varName, case when varName = 'decisionResult' then 0 when varName like '%_callResult' then 2 else case when exists(select * from RulePolicyParameters where clientId = a.clientId  and policyId = a.policyId and varName = b.varName) then 1 else 3 end end

       orden from RulePolicyExecutions a

       inner join RulePolicyExecutionDumps b

             on a.executionId = b.executionId

       where clientId = @clientId and (policyId = @policyid or @policyId = -1)

       and DATEADD(dd, DATEDIFF(dd, 0, a.executionDate), 0) between cast(@from as smalldatetime) and cast(@to as smalldatetime)

       )

       order by orden, varName

 

       open ors

 

       declare @varName varchar(200), @orden smallint

 

       fetch next from ors into @varName, @orden

       while @@ERROR = 0 and @@FETCH_STATUS = 0

       begin

             BEGIN TRY 

                    exec('alter table #tmp add ['+@varName+'] varchar(500)')

             END TRY

             BEGIN CATCH 

             END CATCH

             exec('update #tmp set ['+@varName+'] = isnull((select top 1 case when isnumeric(varvalue) = 1 and varvalue LIKE ''%.%'' then replace(varvalue,''.'','','') else varvalue end varvalue from rulepolicyexecutiondumps with(nolock) where executionid = #tmp.executionid and varname = '''+@varname+'''),'''')')

             fetch next from ors into @varName, @orden

       end

 

       close ors

       deallocate ors

 

      

       select * from #tmp

       drop table #tmp